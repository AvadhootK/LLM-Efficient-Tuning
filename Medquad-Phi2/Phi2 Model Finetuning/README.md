# README
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
