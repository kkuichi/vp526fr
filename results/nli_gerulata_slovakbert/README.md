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
- accuracy
model-index:
- name: nli_gerulata_slovakbert
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE MNLI
      type: glue
      args: mnli
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.7337349397590361
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# nli_gerulata_slovakbert

This model is a fine-tuned version of [gerulata/slovakbert](https://huggingface.co/gerulata/slovakbert) on the GLUE MNLI dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6482
- Accuracy: 0.7337

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

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.7978        | 1.0   | 2500 | 0.6491          | 0.7329   |
| 0.6543        | 2.0   | 5000 | 0.6552          | 0.7478   |
| 0.3346        | 3.0   | 7500 | 0.8042          | 0.7530   |


### Framework versions

- Transformers 5.0.0
- Pytorch 2.10.0+cpu
- Datasets 4.5.0
- Tokenizers 0.22.2
