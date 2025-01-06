# Sys path hacks
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Train script
from datadreamer import DataDreamer
from datadreamer.steps import DataSource
from peft import LoraConfig

from SIVs.datadreamer_lora.luar_utils import get_luar_trainer

with DataDreamer('./SIVs/datadreamer_lora_training/output', verbose=True):
    dataset = DataSource(
        "Training Data",
        data={
            "anchors": ["Apple", "Sky", "Grass", "Carrots", "Sun", "Flamingo"],
            "positives": ["Red", "Blue", "Green", "Orange", "Yellow", "Pink"],
        },
    )
    val_dataset = dataset.take(2)

    trainer = get_luar_trainer()(
        "LUAR Trainer",
        model_name="./rrivera1849/LUAR-MUD",
        peft_config=LoraConfig(),
        trust_remote_code=True,
        force=True,
    )
    trainer.train_with_positive_pairs(
        train_anchors=dataset.output["anchors"],
        train_positives=dataset.output["positives"],
        validation_anchors=val_dataset.output["anchors"],
        validation_positives=val_dataset.output["positives"],
        learning_rate=0.1,
        epochs=5,
        batch_size=8,
        early_stopping_patience=None,
    )