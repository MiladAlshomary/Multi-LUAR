from transformers import PretrainedConfig


import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, PreTrainedModel
from transformers import AutoConfig, AutoModel
import argparse

class LUARConfig(PretrainedConfig):
    model_type = "LUAR"
    
    def __init__(self,
        embedding_size: int = 512,
        use_memory_efficient_attention=False,
        q_bucket_size=512,
        k_bucket_size=1024,
        **kwargs,
    ):
        self.embedding_size = embedding_size
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size
        super().__init__(**kwargs)
        
class MultiLUARsConfig(PretrainedConfig):
    model_type = "MultiLUARs"
    
    def __init__(self,
        embedding_size: int = 512,
        use_memory_efficient_attention=False,
        q_bucket_size=512,
        k_bucket_size=1024,
        sentence_transformer_support=False,
        **kwargs,
    ):
        self.embedding_size = embedding_size
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size
        self.sentence_transformer_support = sentence_transformer_support
        self.hidden_size = embedding_size
        super().__init__(**kwargs)


class SelfAttention(nn.Module):
    """Implements Dot-Product Self-Attention as used in "Attention is all You Need".
    """
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, k, q, v):
        d_k = q.size(-1)
        scores = torch.matmul(k, q.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, v)
    
############################################################
# Self-Attention Mechanisms
############################################################

# Adapted LucidRains impl. of Memory Efficient Attention
# https://github.com/lucidrains/memory-efficient-attention-pytorch

def exists(val):
    return val is not None

def summarize_qkv_chunk(
    q, k, v, 
    mask
):
    """Dot-Product Attention for a chunk of queries, keys, and values.
    """
    weight = torch.einsum('b h i d, b h j d -> b h i j', q, k)
    if exists(mask):
        # HuggingFace masks have to be added:
        weight += mask

    weight_max = weight.amax(dim = -1, keepdim = True).detach()
    weight = weight - weight_max

    exp_weight = weight.exp()
    weighted_value = torch.einsum('b h i j, b h j d -> b h i d', exp_weight, v)

    return exp_weight.sum(dim = -1), weighted_value, rearrange(weight_max, '... 1 -> ...')

checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)

def memory_efficient_attention(
    q, k, v,
    mask = None,
    q_bucket_size = 512,
    k_bucket_size = 1024,
    eps = 1e-8
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # function
    needs_backwards = q.requires_grad or k.requires_grad or v.requires_grad
    summarize_qkv_fn = checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk

    # chunk all the inputs
    q_chunks = q.split(q_bucket_size, dim = -2)
    k_chunks = k.split(k_bucket_size, dim = -2)
    v_chunks = v.split(k_bucket_size, dim = -2)
    mask_chunks = mask.split(k_bucket_size, dim = -1) if exists(mask) else ((None,) * len(k_chunks))

    # loop through all chunks and accumulate
    out = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = []
        weighted_values = []
        weight_maxes = []
        
        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):

            exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_fn(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
            )

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        exp_weights = torch.stack(exp_weights, dim = -1)
        weighted_values = torch.stack(weighted_values, dim = -1)
        weight_maxes = torch.stack(weight_maxes, dim = -1)

        global_max = weight_maxes.amax(dim = -1, keepdim = True)
        renorm_factor = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm_factor
        weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

        all_values = weighted_values.sum(dim = -1)
        all_weights = exp_weights.sum(dim = -1)

        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        out.append(normalized_values)

    return torch.cat(out, dim = -2)


class MemoryEfficientAttention(nn.Module):
    """Memory Efficient Attention: https://arxiv.org/abs/2112.05682
    
       Memory Complexity - O(log n)
       Time Complexity - O(n^2)
    """
    def __init__(
        self,
        *,
        dim = 768,
        heads = 12,
        dim_head = 64,
        memory_efficient = False,
        q_bucket_size = 512,
        k_bucket_size = 1024
    ):
        super().__init__()
        self.heads = heads

        inner_dim = heads * dim_head

        self.key = nn.Linear(dim, inner_dim)
        self.query = nn.Linear(dim, inner_dim)
        self.value = nn.Linear(dim, inner_dim)

        self.memory_efficient = memory_efficient
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        # the following parameters are expected by the HuggingFace
        # implementation of Attention but not used here:
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        h = self.heads

        k = self.key(hidden_states)
        q = self.query(hidden_states)
        v = self.value(hidden_states)

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), 
            (q, k, v)
        )

        out = memory_efficient_attention(
            q, k, v, 
            mask=attention_mask, 
            q_bucket_size=self.q_bucket_size, 
            k_bucket_size=self.k_bucket_size
        )

        out = rearrange(out, 'b h n d -> b n (h d)')

        return (out,)


class MultiLUARs(PreTrainedModel):
    config_class = LUARConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.create_transformer()
        # self.attn_fn = SelfAttention(
        #     config.use_memory_efficient_attention,
        #     config.q_bucket_size,
        #     config.k_bucket_size,
        # )

        self.attn_fn = SelfAttention()
        
        self.layer_linear = nn.ModuleList(
            [nn.Linear(self.hidden_size, config.embedding_size) for _ in range(self.transformer.config.num_hidden_layers + 1)]
        )

        self.sentence_transformer_support = config.sentence_transformer_support

    def create_transformer(self):
        """Creates the Transformer backbone.
        """
        self.transformer = AutoModel.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v1")
        self.hidden_size = self.transformer.config.hidden_size
        self.num_attention_heads = self.transformer.config.num_attention_heads
        self.dim_head = self.hidden_size // self.num_attention_heads

        self.replace_attention()


    def replace_attention(self):
        """Replaces the Transformer's Attention mechanism.
        
           NOTE: This feature has only been tested with the regular 
                 original LUAR SBERT pretrained-model.
        """ 
        attn_fn = {
            "memory_efficient": partial(
                MemoryEfficientAttention, 
                q_bucket_size=32, k_bucket_size=32, 
                heads=self.num_attention_heads, dim_head=self.dim_head,
            ),
        }
        
        for i, layer in enumerate(self.transformer.encoder.layer):
            attention = attn_fn["memory_efficient"]()
            
            state_dict = layer.attention.self.state_dict()
            attention.load_state_dict(state_dict, strict=True)

            self.transformer.encoder.layer[i].attention.self = attention
        
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean Pooling as described in the SBERT paper.
        """
        input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=self.hidden_size).type(token_embeddings.type())
        sum_embeddings = reduce(token_embeddings * input_mask_expanded, 'b l d -> b d', 'sum')
        sum_mask = torch.clamp(reduce(input_mask_expanded, 'b l d -> b d', 'sum'), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_episode_embeddings(self, input_ids, attention_mask, output_attentions=False, document_batch_size=0, average_layers=False):
        """Computes the Author Embedding. 
        """
        B, E, _ = attention_mask.shape
        input_ids = rearrange(input_ids, 'b e l -> (b e) l')
        attention_mask = rearrange(attention_mask, 'b e l -> (b e) l')
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
         
        all_layer_comment_embeddings = []
        all_layer_episode_embeddings = []
        
        for layer_idx, layer_hidden_state in enumerate(outputs['hidden_states']):
            # Mean pooling
            layer_comment_embeddings = self.mean_pooling(layer_hidden_state, attention_mask)
            layer_comment_embeddings = rearrange(layer_comment_embeddings, '(b e) l -> b e l', b=B, e=E)
            all_layer_comment_embeddings.append(layer_comment_embeddings)
            # Attention mechanism and reduce
            layer_episode_embeddings = self.attn_fn(layer_comment_embeddings, layer_comment_embeddings, layer_comment_embeddings)
            layer_episode_embeddings = reduce(layer_episode_embeddings, 'b e l -> b l', 'max')
            # Use the specific linear layer for this transformer layer
            layer_episode_embeddings = self.layer_linear[layer_idx](layer_episode_embeddings)
            # Add a layer dimension
            layer_episode_embeddings = rearrange(layer_episode_embeddings, 'b d -> 1 b d')
            all_layer_episode_embeddings.append(layer_episode_embeddings)

        # Concatenate all layers along the first dimension
        all_layer_episode_embeddings = torch.cat(all_layer_episode_embeddings, dim=0)
        if average_layers:
            all_layer_episode_embeddings = torch.mean(all_layer_episode_embeddings, dim=0)
            
        return all_layer_episode_embeddings

    def rearrange_inputs(self, input_ids, attention_mask):

        seq_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        if seq_length % 32 != 0:
            new_dimension = math.ceil(seq_length/32) * 32
            # Expanding the input_ids to be multiplication of 32
            last_clm = input_ids[:, -1].unsqueeze(1)
            last_clm_expanded = einops.repeat(last_clm, 'n m -> n (repeat m)', repeat=new_dimension-seq_length)
            input_ids = torch.cat([input_ids, last_clm_expanded], axis=1)

            # Expanding the attention_mask to be multiplication of 32
            last_clm = attention_mask[:, -1].unsqueeze(1)
            last_clm_expanded = einops.repeat(last_clm, 'n m -> n (repeat m)', repeat=new_dimension-seq_length)
            attention_mask = torch.cat([attention_mask, last_clm_expanded], axis=1)

        seq_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        
        episode_length = int(seq_length/32)
        
        if episode_length == 0:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)
        else:
            input_ids = input_ids.reshape(batch_size, episode_length, -1)
            attention_mask = attention_mask.reshape(batch_size, episode_length, -1)
            
        return input_ids, attention_mask
        
    def forward(self, input_ids, attention_mask, output_attentions=False, document_batch_size=0, average_layers=False, **kwargs):
        """Calculates a fixed-length feature vector for a batch of episode samples.
        """
        if self.sentence_transformer_support:
            input_ids, attention_mask = self.rearrange_inputs(input_ids, attention_mask)
        
        output = self.get_episode_embeddings(input_ids, attention_mask, output_attentions, document_batch_size, average_layers=average_layers)

        if self.sentence_transformer_support:
            output = rearrange(output, 'l b e-> b l e')
            return output, output, output
        
        else:
            return output

class LUAR(PreTrainedModel):
    """Defines the LUAR model.
    """
    config_class = LUARConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.create_transformer()
        self.attn_fn = SelfAttention(
            config.use_memory_efficient_attention,
            config.q_bucket_size,
            config.k_bucket_size,
        )
        self.linear = nn.Linear(self.hidden_size, config.embedding_size)

    def create_transformer(self):
        """Creates the Transformer backbone.
        """
        self.transformer = AutoModel.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v1")
        self.hidden_size = self.transformer.config.hidden_size
        self.num_attention_heads = self.transformer.config.num_attention_heads
        self.dim_head = self.hidden_size // self.num_attention_heads
        
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean Pooling as described in the SBERT paper.
        """
        input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=self.hidden_size).type(token_embeddings.type())
        sum_embeddings = reduce(token_embeddings * input_mask_expanded, 'b l d -> b d', 'sum')
        sum_mask = torch.clamp(reduce(input_mask_expanded, 'b l d -> b d', 'sum'), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_episode_embeddings(self, input_ids, attention_mask, output_attentions=False, document_batch_size=0):
        """Computes the Author Embedding. 
        """
        B, E, _ = attention_mask.shape

        input_ids = rearrange(input_ids, 'b e l -> (b e) l')
        attention_mask = rearrange(attention_mask, 'b e l -> (b e) l')

        if document_batch_size > 0:
            outputs = {"last_hidden_state": [], "attentions": []}
            for i in range(0, len(input_ids), document_batch_size):
                out = self.transformer(
                    input_ids=input_ids[i:i+document_batch_size],
                    attention_mask=attention_mask[i:i+document_batch_size],
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=output_attentions,
                )
                outputs["last_hidden_state"].append(out["last_hidden_state"])
                if output_attentions:
                    outputs["attentions"].append(out["attentions"])
            outputs["last_hidden_state"] = torch.cat(outputs["last_hidden_state"], dim=0)
            if output_attentions:
                outputs["attentions"] = tuple([torch.cat([x[i] for x in outputs["attentions"]], dim=0) for i in range(len(outputs["attentions"][0]))])
        else:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=False,
                output_attentions=output_attentions,
            )
            
        # at this point, we're embedding individual "comments"
        comment_embeddings = self.mean_pooling(outputs['last_hidden_state'], attention_mask)
        comment_embeddings = rearrange(comment_embeddings, '(b e) l -> b e l', b=B, e=E)

        # aggregate individual comments embeddings into episode embeddings
        episode_embeddings = self.attn_fn(comment_embeddings, comment_embeddings, comment_embeddings)
        episode_embeddings = reduce(episode_embeddings, 'b e l -> b l', 'max')
        
        episode_embeddings = self.linear(episode_embeddings)
        
        if output_attentions:
            return episode_embeddings, outputs["attentions"]

        return episode_embeddings
    
    def forward(self, input_ids, attention_mask, output_attentions=False, document_batch_size=0):
        """Calculates a fixed-length feature vector for a batch of episode samples.
        """
        output = self.get_episode_embeddings(input_ids, attention_mask, output_attentions, document_batch_size)

        return output

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='create trasnformer model',
                    description='')

    parser.add_argument('--output_path') 
    parser.add_argument('--checkpoint_path')      # option that takes a value
    parser.add_argument('--luar', action='store_true', default=False)

    args = parser.parse_args()
    
    #output_path = './reddit_model'
    #checkpoint_path = '/mnt/swordfish-pool2/nikhil/LUAR/src/output/reddit_model/lightning_logs/version_2/checkpoints/epoch=19-step=255100.ckpt'

    # To save our repdocued luar model
    # python saving_transformer_model.py --output_path ../../data/reproduced-luar/ --checkpoint_path /mnt/swordfish-pool2/nikhil/LUAR/src/output/reddit_model/lightning_logs/version_5/checkpoints/epoch\=19-step\=255100.ckpt --luar

    # To save our MultiLUAR
    # python saving_transformer_model.py --output_path ../../data/reproduced-luar/ --checkpoint_path /mnt/swordfish-pool2/nikhil/LUAR/src/output/reddit_model/lightning_logs/version_2/checkpoints/epoch=19-step=255100.ckpt
    
    if args.luar:
        luar_config = LUARConfig()
        pretrained_model = LUAR(luar_config)
        checkpoint = torch.load(args.checkpoint_path)
        pretrained_model.load_state_dict(checkpoint['state_dict'], strict=True)
    
        pretrained_model.save_pretrained(args.output_path)
        luar_config.save_pretrained(args.output_path)
    else:
        luar_config = MultiLUARsConfig()
        pretrained_model = MultiLUARs(luar_config)
        checkpoint = torch.load(args.checkpoint_path)
        pretrained_model.load_state_dict(checkpoint['state_dict'], strict=True)
    
        pretrained_model.save_pretrained(args.output_path)
        luar_config.save_pretrained(args.output_path)
        
    