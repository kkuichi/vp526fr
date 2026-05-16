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
- name: ner-uner_gerulata_slovakbert
  results:
  - task:
      name: Token Classification
      type: token-classification
    dataset:
      name: slovak-nlp/sklep ner-uner
      type: slovak-nlp/sklep
      args: ner-uner
    metrics:
    - name: Precision
      type: precision
      value: 0.721763085399449
    - name: Recall
      type: recall
      value: 0.8238993710691824
    - name: F1
      type: f1
      value: 0.7694566813509545
    - name: Accuracy
      type: accuracy
      value: 0.9783240398963323
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ner-uner_gerulata_slovakbert

This model is a fine-tuned version of [gerulata/slovakbert](https://huggingface.co/gerulata/slovakbert) on the slovak-nlp/sklep ner-uner dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0798
- Precision: 0.7218
- Recall: 0.8239
- F1: 0.7695
- Accuracy: 0.9783

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

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| 0.0318        | 1.0   | 1061 | 0.0803          | 0.7218    | 0.8239 | 0.7695 | 0.9783   |
| 0.0192        | 2.0   | 2122 | 0.0960          | 0.7250    | 0.8789 | 0.7946 | 0.9756   |
| 0.0126        | 3.0   | 3183 | 0.0812          | 0.7812    | 0.8475 | 0.8130 | 0.9812   |


### Framework versions

- Transformers 5.0.0
- Pytorch 2.10.0+cpu
- Datasets 4.5.0
- Tokenizers 0.22.2
