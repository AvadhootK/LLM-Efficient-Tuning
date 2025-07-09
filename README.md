# LLM-Efficient-Tuning: Dynamic LoRA Rank Optimization for Efficient Fine-Tuning

This repository contains experiments on a dynamic LoRA rank optimization framework for parameter-efficient fine-tuning of large language models (LLMs). Fine-tuning LLMs is costly, and while LoRA improves efficiency by training low-rank matrices, selecting the right rank is challenging. The new approach leverages intrinsic dimensionality (ID) analysis of hidden representations to determine optimal LoRA ranks layer-wise, striking a balance between model expressivity and computational efficiency.

## Key Features & Contributions

- Computed intrinsic dimensionality (ID) of transformer activations using scikit-dimension estimators (MLE, DANCo, Fisher Separability, TwoNN, TLE, PCA, kNN).
- Mapped ID estimates to LoRA ranks dynamically during fine-tuning for different LLMs.
- Ran controlled fine-tuning experiments with fixed vs. adaptive ranks using Hugging Face's PEFT and OpenDelta libraries.
- Benchmarked the impact of rank selection on performance and training cost across datasets like GLUE, SQuAD v2, TruthfulQA, MedQuAD.
- Performed truthfulness and hallucination analysis using Local Intrinsic Dimension (LID) as a proxy for factual drift in generation.
- Evaluated inference-time optimizations like quantization and model pruning.

## Benchmarking & Metrics
- BLEU, ROUGE, BERTScore, F1
- Local Intrinsic Dimension (LID)
- "LLM-as-a-Judge" metrics for factual consistency

## Models Used

- LLaMA 2
- Phi-2
- BERT
- DeBERTa
- RoBERTa

## Datasets Used

- SQuAD v2 – QA with unanswerable questions
- TruthfulQA – Hallucination and factuality
- MedQuAD – Domain-specific medical QA
- GLUE / SuperGLUE – General NLP benchmarks
- SP – Sentiment propagation dataset

## Experiment Outputs

## Results of Experiments on Selecting Best Intrinsic Dimension Estimator

### 1. Effect of Noise on ID Estimator

![image](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/8ec54dbb-d968-4afc-93ec-b796afee7d7c)

### 2. Effect of Number of Samples on ID Estimator

![image](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/b4ff1655-6d0e-4dd9-99b2-aa9d78826792)

### 3. Matrix Entropy Evaluation
### Phi2 Model Experiments

a. Matrix Entropy Visualization

![image](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/25ba1020-29b6-4e74-a87c-d38cf6413a8f)

b. Correlation Metric: 
#### Pearson's correlation coefficient between matrix entropy and Intrinsic dimensions 
- Pearson's Correlation for KNN: 0.045
- Pearson's Correlation for MLE: 0.952
- Pearson's Correlation for 2NN: 0.678
- Pearson's Correlation for Fisher Separability: 0.640
- Pearson's Correlation for Correlation Dimension: 0.722
- Pearson's Correlation for TLE: 0.959
- Pearson's Correlation for PCA: 0.546
- Pearson's Correlation for Persistent Homology: 0.951
- Pearson's Correlation for Mean: 0.937

### DistilBERT Model Experiments

a. Matrix Entropy Visualization

![image](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/83b232cf-1e5d-47ca-8033-daeb026baed5)


b. Correlation Metric:
#### Pearson's correlation coefficient between matrix entropy and Intrinsic dimensions
#### Including Embedding Layer
  - Pearson's correlation for TwoNN: -0.211
  - Pearson's correlation for Maximum Likelihood: -0.071
  - Pearson's correlation for Correlation Dimension: 0.871
  - Pearson's correlation for Tight Local ID: 0.010
  - Pearson's correlation for Persistent Homology: 0.915
 ##### Excluding Embedding Layer
  - Pearson's correlation for TwoNN: 0.775
  - Pearson's correlation for Maximum Likelihood: 0.968
  - Pearson's correlation for Correlation Dimension: 0.360
  - Pearson's correlation for Tight Local ID: 0.908
  - Pearson's correlation for Persistent Homology: 0.952
  
#### Kendall's rank correlation between matrix entropy and Intrinsic dimensions
#### Including Embedding Layer
- Kendall's rank correlation for TwoNN: -0.048
- Kendall's rank correlation for Maximum Likelihood: 0.333
- Kendall's rank correlation for Correlation Dimension: 0.333
- Kendall's rank correlation for Tight Local ID: 0.238
- Kendall's rank correlation for Persistent Homology: 0.810
#### Excluding Embedding Layer
- Kendall's rank correlation for TwoNN: 0.333
- Kendall's rank correlation for Maximum Likelihood: 0.867
- Kendall's rank correlation for Correlation Dimension: 0.067
- Kendall's rank correlation for Tight Local ID: 0.733
- Kendall's rank correlation for Persistent Homology: 0.733

#### Spearman’s Rank Correlation between matrix entropy and Intrinsic dimensions
#### Including Embedding Layer
- Spearman’s Rank Correlation for TwoNN: 0.000
- Spearman’s Rank Correlation for Maximum Likelihood: 0.214
- Spearman’s Rank Correlation for Correlation Dimension: 0.536
- Spearman’s Rank Correlation for Tight Local ID: 0.143
- Spearman’s Rank Correlation for Persistent Homology: 0.929
#### Excluding Embedding Layer
- Spearman’s Rank Correlation for TwoNN: 0.600
- Spearman’s Rank Correlation for Maximum Likelihood: 0.943
- Spearman’s Rank Correlation for Correlation Dimension: 0.257
- Spearman’s Rank Correlation for Tight Local ID: 0.829
- Spearman’s Rank Correlation for Persistent Homology: 0.886

## Phi-2 Model Pruning and ID Analysis

This section presents findings from pruning the Phi-2 model based on intrinsic dimensionality.

1.  **Phi-2 Model Pruning:**
    Phi-2 Model pruning to include only layers till Minimum Intrinsic Dimension Layer.

2.  **ID Profile Analysis Before and After Fine-Tuning:**
    Analyzing ID Profile before and after fine-tuning the pruned Phi-2 model. This helps in understanding the impact of pruning on the model's representational capacity.

    ![Phi-2 ID Profile Analysis](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/22c23187-3dcf-4101-8f74-703fb6c747bc)

## Intrinsic Dimensionality Analysis on Phi-2

This section details the intrinsic dimensionality analysis conducted on the Phi-2 model, particularly on the MedQuAD dataset.

1.  **Intrinsic Dimension Profile of Phi-2 on MedQuAD:**
    Presents the intrinsic dimension profile of the Phi-2 model when evaluated on the MedQuAD dataset.

    ![Phi-2 MedQuAD ID Profile](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/8f069e39-3bdc-40ed-ad3d-b2525aff3f78)

2.  **Computed Batched Intrinsic Dimensions for Phi-2 on MedQuAD:**
    Details the computation of batched Intrinsic Dimensions for the Phi-2 model on the MedQuAD dataset, including the evaluation of ID using mean and standard deviation across all batches.

3.  **Intrinsic Dimension Profile Analysis for Different ID Estimators on Phi-2 (MedQuAD):**
    Analyzes the intrinsic dimension profile using various ID estimators specifically for the Phi-2 model on the MedQuAD dataset.

    ![Phi-2 MedQuAD ID Estimators 1](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/437384a5-ff1f-45f6-8015-3c6c100fceba)
    ![Phi-2 MedQuAD ID Estimators 2](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/5421cd59-6758-41e2-a3a3-b12fc6b8d5be)

## Fine-tuned vs. Base Model ID Profile Comparison

This section compares the intrinsic dimension profiles of the base Phi-2 model with its fine-tuned counterpart.

1.  **Analyzing Base Phi-2 Model ID vs. Fine-tuned Phi-2 Model ID:**
    A comparative analysis illustrating how fine-tuning impacts the intrinsic dimensionality profile of the Phi-2 model's hidden representations.

    ![Fine-tuned vs. Base Phi-2 ID Profile](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/ac5b67c4-a23c-48f8-8f64-fca5807262db)

## Intrinsic Dimensionality Analysis on DistilBERT for Sentiment Analysis

These outputs demonstrate the computation and analysis of intrinsic dimensions for DistilBERT on a sentiment analysis task.

1.  **Batched Intrinsic Dimensions for DistilBERT:**
    Computed batched Intrinsic Dimensions for DistilBERT on sentiment analysis task.

    ![Intrinsic Dimension Batches 1](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/b5d61fe3-5cdc-4c39-a58c-74ed49674a1e)

2.  **Evaluation of ID using Mean and Standard Deviation:**
    Evaluation of ID using mean and standard deviation across all batches.

    ![Intrinsic Dimension Batches 2](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/e9e978ec-60eb-4376-8fe7-95602d6b64f5)

3.  **Analysis of Confidence Interval:**
    Analysis of Confidence Interval for the intrinsic dimensionality estimates.

4.  **Intrinsic Dimension Profile Analysis for Different ID Estimators:**
    Intrinsic dimension profile analysis comparing various ID estimators.

    ![ID Profile Analysis](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/722a0c01-2d49-4498-9b83-a79c67049e92)

5.  **Multiple Trials for ID Estimators:**
    Running 10 trials for all ID estimators to ensure robustness and consistency of results.

6.  **Local Density Calculation:**
    Demonstrates the calculation of local density within the hidden representations.

    ![Local Density Calculation](https://github.com/abdessalam-eddib/llm_experiments/assets/72447002/de60021c-35b8-4ba4-8bc3-47d896fc8806)
