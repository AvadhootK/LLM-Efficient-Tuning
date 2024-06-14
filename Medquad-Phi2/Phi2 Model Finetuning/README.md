# README
### Intrinsic Dimentions - Phi2
1. Intrinsic Dimension profile of Phi2 model on Medquad dataset. 

### Finetune Phi2 QLoRA
1. Fine-tuning Phi2 model
  - Dataset: MedQuad
  - Method: QLoRA
  - LoRA Ranks: 1, 14 (maximum ID), ID as ranks of adapter layers, 64

### Finetune Phi2 QLoRA with alpha
1. Fine-tuning Phi2 model
  - Dataset: MedQuad
  - Method: QLoRA
  - LoRA Ranks: 1, 6 (minimum ID), 14 (maximum ID), ID as ranks of adapter layers, 64
  - Scaling factor (alpha): 8 * ranks
2. Deriving 10 results from the fine-tuned model for each rank.

### Finetune model ID Profile + base model inference
1. Analyzing base phi2 model ID vs fine-tuned phi2 model ID

### computing-batch-ids
1. Computed batched Intrinsic Dimensions for Phi2 model on MedQuad dataset.
2. Evaluation of ID using mean and standard deviation across all batches.
