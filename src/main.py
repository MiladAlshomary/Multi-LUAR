# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import random

import numpy as np
import pytorch_lightning as pt
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from arguments import create_argument_parser
from models.transformer import Transformer
from utilities.file_utils import Utils as utils
from transformers import AutoModel, AutoTokenizer

LUAR_PATH =  "/mnt/swordfish-pool2/nikhil/LUAR/pretrained_weights/LUAR-MUD/"

def main(params):
    # set random seeds reproduceability
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    # weirdness with HuggingFace tokenizer when processing things in parallel
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy('file_system')
        
    # create experiment_dir and load model
    experiment_dir = os.path.join(utils.output_path, params.experiment_id)
    experiment_dir = utils.path_exists(experiment_dir, True)
    
    model = Transformer(params)

    # # Step 2: Load pretrained weights from AutoModel
    # pretrained_model = AutoModel.from_pretrained(LUAR_PATH, trust_remote_code=True)

    # # Step 3: Extract the state_dict from the pretrained model
    # pretrained_state_dict = pretrained_model.state_dict()

    # # Step 4: Load the pretrained weights into your Transformer model
    # # Ensure strict=False to avoid mismatch errors if there are additional or missing keys
    # model.load_state_dict(pretrained_state_dict, strict=False)

    # compute validation if needed, otherwise just skip it and save 
    # every `period` checkpoints
    if params.validate:
        limit_val_batches = 1.0
        checkpoint_callback = ModelCheckpoint(
            monitor="val_R@8", 
            mode="max"
        )
    else:
        limit_val_batches = 0.0
        checkpoint_callback = ModelCheckpoint(
            monitor=None, 
            save_top_k=-1, 
            every_n_epochs=params.period
        )

    # load checkpoint if necessary
    resume_from_checkpoint = None
    if params.load_checkpoint:
        # get the latest checkpoint
        version = "version_0" if params.version is None else params.version
        path = os.path.join(experiment_dir, params.log_dirname, version, 'checkpoints', '*.ckpt')
        resume_from_checkpoint = glob.glob(path)[0]
        print("Checkpoint: {}".format(resume_from_checkpoint))

        checkpoint = torch.load(resume_from_checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    logger = TensorBoardLogger(experiment_dir, name=params.log_dirname, version=params.version)
    trainer = pt.Trainer(
        default_root_dir=experiment_dir,
        max_epochs=params.num_epoch,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator='gpu',  # Specify GPU as the accelerator
        devices=[0],  # Use appropriate GPU count
        strategy='dp' if len(params.gpus) > 1 else None,
        precision=params.precision,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=params.validate_every if params.validate else 1,
        num_sanity_val_steps=0,
    )

    if params.do_learn:
        trainer.fit(model)

    if params.evaluate:
        trainer.test(model)

if __name__ == "__main__":
    parser = create_argument_parser()
    main(parser.parse_args())
