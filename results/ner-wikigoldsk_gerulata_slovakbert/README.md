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
- name: ner-wikigoldsk_gerulata_slovakbert
  results:
  - task:
      name: Token Classification
      type: token-classification
    dataset:
      name: slovak-nlp/sklep ner-wikigoldsk
      type: slovak-nlp/sklep
      args: ner-wikigoldsk
    metrics:
    - name: Precision
      type: precision
      value: 0.8901098901098901
    - name: Recall
      type: recall
      value: 0.8966789667896679
    - name: F1
      type: f1
      value: 0.8933823529411764
    - name: Accuracy
      type: accuracy
      value: 0.9841744744508306
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ner-wikigoldsk_gerulata_slovakbert

This model is a fine-tuned version of [gerulata/slovakbert](https://huggingface.co/gerulata/slovakbert) on the slovak-nlp/sklep ner-wikigoldsk dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0622
- Precision: 0.8901
- Recall: 0.8967
- F1: 0.8934
- Accuracy: 0.9842

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
| 0.1138        | 1.0   | 586  | 0.0841          | 0.8568    | 0.8773 | 0.8669 | 0.9805   |
| 0.0769        | 2.0   | 1172 | 0.0672          | 0.8896    | 0.8921 | 0.8908 | 0.9833   |
| 0.0447        | 3.0   | 1758 | 0.0622          | 0.8901    | 0.8967 | 0.8934 | 0.9842   |


### Framework versions

- Transformers 5.0.0
- Pytorch 2.10.0+cpu
- Datasets 4.5.0
- Tokenizers 0.22.2
