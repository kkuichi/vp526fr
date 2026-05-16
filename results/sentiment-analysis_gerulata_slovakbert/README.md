---
library_name: transformers
license: mit
base_model: gerulata/slovakbert
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: sentiment-analysis_gerulata_slovakbert
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sentiment-analysis_gerulata_slovakbert

This model is a fine-tuned version of [gerulata/slovakbert](https://huggingface.co/gerulata/slovakbert) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0822
- Accuracy: 0.9751

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
| 0.1133        | 1.0   | 445  | 0.0823          | 0.9751   |
| 0.0305        | 2.0   | 890  | 0.1090          | 0.9751   |
| 0.0277        | 3.0   | 1335 | 0.1194          | 0.9732   |


### Framework versions

- Transformers 5.0.0
- Pytorch 2.10.0+cpu
- Datasets 4.5.0
- Tokenizers 0.22.2
