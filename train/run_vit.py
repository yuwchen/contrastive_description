import pandas as pd
import os
import shutil
from tqdm import tqdm
from config import *
from utils import *

from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)
import torch

from accelerate import Accelerator
accelerator = Accelerator()


torch.set_num_threads(128)


def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

dataset = load_dataset(hparams['dataset_root_path'], data_dir=hparams['data_dir'])
train_ds, test_ds = dataset['train'], dataset['test']
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

id2label = get_id2label(train_ds)
label2id = get_label2id(id2label)

## Preprocess the data
processor = ViTImageProcessor.from_pretrained(hparams['model_name'])

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
)

# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

def train(args):

    ### Define the model
    model = ViTForImageClassification.from_pretrained(hparams['model_name'],
                                                    id2label=id2label,
                                                    label2id=label2id,
                                                    ignore_mismatched_sizes=True)

    if not hparams['use_cpu'] and torch.cuda.is_available():
        model = model.to(hparams['device'])

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor
    )

    trainer = accelerator.prepare(trainer)

    # start_training
    trainer.train()
    best_acc = trainer.evaluate()['eval_accuracy']

    # save the best model based on eval accuracy
    model_path = hparams['save_best_model_path'] + str(best_acc)
    # model_path = f'/local/data/xuanming/vit_base_32_in21k/vit_{best_acc:.2f}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    trainer.save_state()
    trainer.save_model(model_path)


def evaluate(args):
    model = ViTForImageClassification.from_pretrained(hparams['save_best_model_path'],
                                                      id2label=id2label,
                                                      label2id=label2id,
                                                      ignore_mismatched_sizes=True)
    
    if not hparams['use_cpu'] and torch.cuda.is_available():
        model = model.to(hparams['device'])

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor
    )

    trainer = accelerator.prepare(trainer)
    outputs = trainer.predict(test_ds)
    print(outputs.metrics)


if __name__ == '__main__':
    args = TrainingArguments(
        output_dir=hparams['output_dir'],
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=hparams['learning_rate'],
        per_device_train_batch_size=hparams['per_device_train_batch_size'],
        per_device_eval_batch_size=hparams['per_device_eval_batch_size'],
        num_train_epochs=hparams['num_train_epochs'],
        weight_decay=hparams['weight_decay'],
        load_best_model_at_end=True,
        metric_for_best_model=hparams['metric_name'],
        logging_dir='./training_logs',
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=hparams['logging_steps'],
        report_to="wandb",
        run_name=hparams['run_name'],
        use_cpu=hparams['use_cpu'],
    )

    if hparams['do_train']:
        train(args)
    else:
        evaluate(args)