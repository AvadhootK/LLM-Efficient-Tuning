# README

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
