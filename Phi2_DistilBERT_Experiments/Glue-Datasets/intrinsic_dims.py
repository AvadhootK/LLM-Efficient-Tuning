# # Intrinsic Dimension Estimation via Persistent Homology

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass, field
from typing import Optional
import torch
import gc 
import time
from transformers import HfArgumentParser
import os
import sys 
import pickle
import skdim

# define Glue tasks and corresponding prompt formats
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-m": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# define dataset arguments 
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))

def main():
    parser = HfArgumentParser(DataTrainingArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, = parser.parse_args_into_dataclasses()
    
    # standardize when dealing with different variations of mnli
    if "mnli" in data_args.task_name:
        data_args.task_name = "mnli"

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_and_score(embeddings, n, k=8, hat_n=40, J=7):
        """
        For various sample sizes, compute the median persistent score across J samples.
        
        Parameters:
        - embeddings (numpy.ndarray): A matrix where each row is an embedding.
        - n (int): Total number of embeddings.
        - k (int): Number of different sample sizes.
        - hat_n (int): A parameter for determining sample sizes.
        - J (int): Number of samples for each sample size.
        
        Returns:
        - list: List of sample sizes.
        - list: List of corresponding median persistent scores.
        """
        scores = []
        sizes = [(i - 1) * (n - hat_n) // k + hat_n for i in range(1, k + 1)]
        
        for size in sizes:
            subset_scores = [compute_persistent_score(embeddings[torch.randperm(n)[:size]])
                            for _ in range(J)]
            scores.append(torch.median(torch.tensor(subset_scores)).item())
        
        return sizes, scores

    def compute_persistent_score(embeddings):
        """
        Compute the persistent score for a subset of embeddings using the sum of edge weights in the MST.
        
        Parameters:
        - embeddings (numpy.ndarray): A matrix where each row is an embedding.
        
        Returns:
        - float: The persistent score for the embeddings.
        """
        dist_matrix = distance_matrix(embeddings, embeddings)
        mst = minimum_spanning_tree(dist_matrix)
        return mst.sum()

    def estimate_dimension(sizes, scores):
        """
        Estimate the intrinsic dimension of the data using linear regression on log-transformed sizes and scores.
        
        Parameters:
        - sizes (list): List of sample sizes.
        - scores (list): List of corresponding median persistent scores.
        
        Returns:
        - float: Estimated dimension of the data.
        """
        log_sizes = np.log(sizes).reshape(-1, 1)
        log_scores = np.log(scores)

        reg = LinearRegression().fit(log_sizes, log_scores)
        slope = reg.coef_[0]
        
        return 1 / (1 - slope)

    def estimate_sequence_dimension(embeddings, runs=20):
        """
        Estimate the intrinsic dimension of the hidden state by repeatedly sampling subsets of its tokens, 
        computing their persistent scores, and then using linear regression on the log-transformed values.
        
        Parameters:
        - embeddings (numpy.array): The embeddings for which the dimension needs to be estimated.
        - runs (int): Number of runs with different random seeds. Default is 20.
        
        Returns:
        - float: Estimated dimension of the embeddings.
        """
        n = embeddings.shape[0]
        
        slopes = []
        for _ in range(runs):
            sizes, scores = sample_and_score(embeddings, n)
            log_sizes = np.log(sizes).reshape(-1, 1)
            log_scores = np.log(scores)
            
            reg = LinearRegression().fit(log_sizes, log_scores)
            slopes.append(reg.coef_[0])
        
        kappa_F = np.mean(slopes)
        return 1 / (1 - kappa_F)
    ## End of code by AmelieSchreiber

    def plot_intrinsic_dims(intrinsic_dims, std=None, num_batches=None, file_path="intrinsic_dims.png"):
        """
        Plots the intrinsic dimensions of hidden states with an optional confidence interval.

        Parameters:
        intrinsic_dims (list or numpy array): A list or array containing the intrinsic dimensions for each hidden state.
        std (list or numpy array, optional): A list or array containing the standard deviations of the intrinsic dimensions for each hidden state. Default is None.
        num_batches (int, optional): The number of batches used to compute the intrinsic dimensions. This is used to calculate the 95% confidence interval if std is provided. Default is None.
        file_path (str, optional): The file path where the plot will be saved. Default is "intrinsic_dims.png".

        Returns:
        None

        The function creates and saves a plot ("intrinsic_dims.png") with the following features:
        - Line plot of intrinsic dimensions with markers for each hidden state.
        - 95% confidence interval as a shaded area if std and num_batches are provided.
        - A red marker indicating the minimum intrinsic dimension with its value annotated.
        """
        min_id = np.argmin(intrinsic_dims)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(intrinsic_dims) + 1), intrinsic_dims, marker="o")

        if std is not None and num_batches is not None:
            plt.fill_between(
                range(1, len(intrinsic_dims) + 1),
                intrinsic_dims - 1.96 * std / np.sqrt(num_batches),
                intrinsic_dims + 1.96 * std / np.sqrt(num_batches),
                alpha=0.3,
                label="95% CI",
            )

        plt.plot(min_id + 1, intrinsic_dims[min_id], "ro", label="Minimum ID")
        plt.annotate(
            f"{intrinsic_dims[min_id]:.2f}",
            (min_id + 1, intrinsic_dims[min_id]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.xlabel("Hidden state")
        plt.ylabel("Intrinsic dimension")
        plt.title("Intrinsic dimensions of hidden states")
        plt.grid()
        plt.legend(loc="best")
        plt.savefig(file_path)

                
    # Aadaptive ranks and alphas
    def set_adaptive_ranks_alphas(id_path, alpha_rank_ratio=32):
        """
        Sets the adaptive ranks and alphas based on the intrinsic dimensions of the hidden states -- Phi2.

        Parameters:
        id_path (str): The path to the file containing the intrinsic dimensions of the hidden states.
        ALPHA_RANK_RATIO (int, optional): The ratio of alpha to rank. Default is 1.

        Returns:
        None

        The function loads the intrinsic dimensions of the hidden states from the saved file "final_ids.npy" and rounds them.
        It then creates dictionaries for the rank pattern and alpha pattern for the adaptive LoRA model.
        The dictionaries are saved as "rank_pattern.pkl" and "alpha_pattern.pkl" respectively.
        """
        ranks = np.round(np.load(id_path)).astype(int)[1:]

        rank_pattern = {}
        for i, rank in enumerate(ranks):
            rank_pattern[f"model.model.layers[{i}].self_attn.q_proj"] = int(rank)
            rank_pattern[f"model.model.layers[{i}].self_attn.k_proj"] = int(rank)
            rank_pattern[f"model.model.layers[{i}].self_attn.v_proj"] = int(rank)
            rank_pattern[f"model.model.layers[{i}].self_attn.o_proj"] = int(rank)

        alpha_pattern = {}
        for i, rank in enumerate(ranks):
            alpha_pattern[f"model.model.layers[{i}].self_attn.q_proj"] = int(rank * alpha_rank_ratio) 
            alpha_pattern[f"model.model.layers[{i}].self_attn.k_proj"] = int(rank * alpha_rank_ratio) 
            alpha_pattern[f"model.model.layers[{i}].self_attn.v_proj"] = int(rank * alpha_rank_ratio) 
            alpha_pattern[f"model.model.layers[{i}].self_attn.o_proj"] = int(rank * alpha_rank_ratio) 

        return rank_pattern, alpha_pattern

    def get_total_budget(rank_pattern):
        """
        Calculates the total budget required for the adaptive LoRA model.

        Parameters:
        rank_pattern (dict): A dictionary containing the ranks of the hidden states.

        Returns:
        int: The total budget required for the adaptive LoRA model.
        """
        return sum(rank_pattern.values())

    def get_mean_rank(rank_pattern):
        """
        Calculates the mean rank of the hidden states.

        Parameters:
        rank_pattern (dict): A dictionary containing the ranks of the hidden states.

        Returns:
        float: The mean rank of the hidden states.
        """
        return np.mean(list(rank_pattern.values()))

    # Clearing CUDA Cache
    def clear_cache():
        """
        Clears the CUDA cache and performs garbage collection to free up memory.

        This function performs the following steps:
        1. Clears the CUDA cache using torch.cuda.empty_cache().
        2. Performs garbage collection using gc.collect().
        3. Pauses execution for 5 seconds using time.sleep(5) to ensure memory is freed.

        Returns:
        None

        Example usage:
        clear_cache()
        """
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(5)

    # Load the embeddings
    embeddings = np.load(f"./dataset_activations/activations_full_{data_args.task_name}.npy")
    # for i, element in enumerate(embeddings):
    #     print(f"Element {i}: Type: {type(element)}, Length: {len(element)}")
    n = embeddings.shape[0]
    # Estimate the intrinsic dimension
    intrinsic_dims = []
    for i in range(n):
        X = embeddings[i]
        # Persistent homology
        # id_estimate = estimate_sequence_dimension(X)
        # TwoNN estimator
        nn_estimator = skdim.id.TwoNN()
        id_estimate = nn_estimator.fit_transform(X)
        intrinsic_dims.append(id_estimate)

    print(f"Estimated Intrinsic Dimension: {intrinsic_dims}")
    # plot_intrinsic_dims(intrinsic_dims[1:])

    import pickle
    # File path to save the pickle file
    intrinsic_dims_file_path = './intrinsic_dims'

    os.makedirs(intrinsic_dims_file_path, exist_ok=True)
    np.save(
        f"{intrinsic_dims_file_path}/ID_{data_args.task_name}.npy",
        intrinsic_dims,
    )

    print(f'Intrinsic dimensions saved to {intrinsic_dims_file_path}/ID_{data_args.task_name}.npy')
    
    rank_pattern, alpha_pattern = set_adaptive_ranks_alphas(f"{intrinsic_dims_file_path}/ID_{data_args.task_name}.npy")
    print("Rank pattern",rank_pattern)
    print("Alpha pattern",alpha_pattern)

    rank_pattern_file_path = './rank_pattern'
    os.makedirs(rank_pattern_file_path, exist_ok=True)
    with open(f"{rank_pattern_file_path}/rank_pattern_{data_args.task_name}.pkl", 'wb') as file:
        pickle.dump(rank_pattern, file)

    alpha_pattern_file_path = './alpha_pattern'
    os.makedirs(alpha_pattern_file_path, exist_ok=True)
    with open(f"{alpha_pattern_file_path}/alpha_pattern_{data_args.task_name}.pkl", 'wb') as file:
        pickle.dump(alpha_pattern, file)

    mean_rank_path = './mean_rank'
    mean_rank = get_mean_rank(rank_pattern)
    print("Mean rank",mean_rank)
    os.makedirs(mean_rank_path, exist_ok=True)
    with open(f"{mean_rank_path}/mean_rank_{data_args.task_name}.pkl", 'wb') as file:
        pickle.dump(mean_rank, file)

if __name__ == "__main__":
    main() 