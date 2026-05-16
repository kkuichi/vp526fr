---
library_name: transformers
language:
- en
license: mit
base_model: gerulata/slovakbert
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- spearmanr
model-index:
- name: sts_gerulata_slovakbert
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE STSB
      type: glue
      args: stsb
    metrics:
    - name: Spearmanr
      type: spearmanr
      value: 0.8583273933048574
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sts_gerulata_slovakbert

This model is a fine-tuned version of [gerulata/slovakbert](https://huggingface.co/gerulata/slovakbert) on the GLUE STSB dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6572
- Pearson: 0.8593
- Spearmanr: 0.8583
- Combined Score: 0.8588

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Pearson | Spearmanr | Combined Score |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:---------:|:--------------:|
| 0.7827        | 1.0   | 701  | 0.7128          | 0.8328  | 0.8366    | 0.8347         |
| 0.4378        | 2.0   | 1402 | 0.6760          | 0.8585  | 0.8586    | 0.8586         |
| 0.3235        | 3.0   | 2103 | 0.6572          | 0.8593  | 0.8583    | 0.8588         |


### Framework versions

- Transformers 5.0.0
- Pytorch 2.10.0+cpu
- Datasets 4.5.0
- Tokenizers 0.22.2
