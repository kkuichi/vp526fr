---
library_name: transformers
license: mit
base_model: gerulata/slovakbert
tags:
- generated_from_trainer
datasets:
- slovak-nlp/sklep
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: pos_gerulata_slovakbert
  results:
  - task:
      name: Token Classification
      type: token-classification
    dataset:
      name: slovak-nlp/sklep pos
      type: slovak-nlp/sklep
      args: pos
    metrics:
    - name: Precision
      type: precision
      value: 0.9772247996625896
    - name: Recall
      type: recall
      value: 0.9759076741639289
    - name: F1
      type: f1
      value: 0.9765657928011463
    - name: Accuracy
      type: accuracy
      value: 0.9806092008164548
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# pos_gerulata_slovakbert

This model is a fine-tuned version of [gerulata/slovakbert](https://huggingface.co/gerulata/slovakbert) on the slovak-nlp/sklep pos dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0980
- Precision: 0.9772
- Recall: 0.9759
- F1: 0.9766
- Accuracy: 0.9806

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

| Training Loss | Epoch | Step | Accuracy | F1     | Validation Loss | Precision | Recall |
|:-------------:|:-----:|:----:|:--------:|:------:|:---------------:|:---------:|:------:|
| 0.0907        | 1.0   | 1061 | 0.9743   | 0.9707 | 0.1364          | 0.9712    | 0.9702 |
| 0.0542        | 2.0   | 2122 | 0.9795   | 0.9756 | 0.1006          | 0.9765    | 0.9747 |
| 0.0338        | 3.0   | 3183 | 0.9802   | 0.9761 | 0.0989          | 0.9768    | 0.9753 |


### Framework versions

- Transformers 5.0.0
- Pytorch 2.10.0+cpu
- Datasets 4.5.0
- Tokenizers 0.22.2
