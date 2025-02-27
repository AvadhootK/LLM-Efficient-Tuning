from datasets import load_from_disk
from typing import Optional
import os
import sys 
import random
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoConfig, 
    AutoTokenizer, 
    PretrainedConfig,
    AutoModelForSequenceClassification
)
import torch 
from tqdm import tqdm
import time
import numpy as np

# path to save the processed datasets
save_path = "./processed_datasets"

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
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # standardize when dealing with different variations of mnli
    if "mnli" in data_args.task_name:
        data_args.task_name = "mnli"

    # Load the preprocessed raw_datasets from disk
    raw_datasets = load_from_disk(os.path.join(training_args.output_dir, save_path, data_args.task_name))

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=True if model_args.token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.token else None,
    )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}


    if data_args.max_seq_length > tokenizer.model_max_length:
        print(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    # tokenize dataset
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )

    # prepare train dataset
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    print(train_dataset[0])
    # prepare validation dataset
    if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    val_dataset = raw_datasets["validation"]
    if data_args.max_val_samples is not None:
        max_val_samples = min(len(val_dataset), data_args.max_val_samples)
        val_dataset = val_dataset.select(range(max_val_samples))

    # prepare test dataset
    if "test" not in raw_datasets and "test_matched" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset")
    test_dataset = raw_datasets["test"]
    if data_args.max_test_samples is not None:
        max_test_samples = min(len(test_dataset), data_args.max_test_samples)
        test_dataset = test_dataset.select(range(max_test_samples))

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     print(f"Sample {index} of the training set: {train_dataset[index]}.")
        
    ####################################################################################################
    # Determine maximum batch size 
    # def get_batch_size(
    #     model,
    #     tokenizer,
    #     num_tokens,
    #     dataset_size,
    #     max_batch_size,
    #     device,
    # ):
    #     """
    #     Determines the optimal batch size for a given model and device, based on
    #     memory constraints and dataset size.

    #     This function increments the batch size exponentially until the maximum
    #     batch size or dataset size limit is reached, or a CUDA out of memory error
    #     occurs, in which case the batch size is halved.

    #     Args:
    #         model (AutoModel): The model to be evaluated.
    #         num_tokens (int): The maximum number of tokens per sample.
    #         dataset_size (int): The total number of samples in the dataset.
    #         max_batch_size (int or None): The maximum batch size allowed. If None,
    #                                     the batch size is not limited by this parameter.
    #         device (torch.device): The device on which the model is to be run.

    #     Returns:
    #         int: The optimal batch size determined by the function.
    #     """

    #     model.to(device)
    #     batch_size = 2

    #     while True:
    #         print(f"Current batch size: {batch_size}")
    #         if batch_size == 0:
    #             print("Reduce sequence length . . .")
    #             break
    #         if max_batch_size is not None and batch_size > max_batch_size:
    #             batch_size = max_batch_size
    #             break

    #         if batch_size >= dataset_size:
    #             batch_size = batch_size // 2
    #             break

    #         try:
    #             for _ in range(10):
    #                 input_shape = (batch_size, num_tokens)
    #                 max_token_id = len(tokenizer.get_vocab()) - 1
    #                 inputs = torch.randint(
    #                     0, max_token_id, input_shape, dtype=torch.long, device=device
    #                 )
    #                 _ = model(inputs)
    #             batch_size *= 2

    #         except RuntimeError as e:
    #             if "CUDA out of memory" in str(e):
    #                 print("CUDA out of memory. Reducing batch size . . .")
    #                 batch_size = batch_size // 2
    #                 return batch_size
    #             else:
    #                 raise e

    #     # del model
    #     # torch.cuda.empty_cache()

    #     return batch_size
    
    # dataset_size = len(train_dataset)
    # device = torch.device("cuda")
    # optimal_batch_size = get_batch_size(model, tokenizer, data_args.max_seq_length, dataset_size, None, device)
    # print("Size of training data of glue",data_args.task_name, dataset_size)
    # print("Optimal batch size for glue",data_args.task_name, optimal_batch_size)

    # ##################################################################################################################
    # # Compute hidden states
    # num_data = 40
    # per_batch = 4
    # number_batches =  num_data // per_batch

    # device = torch.device("cuda")
    # model.to(device)

    # # Collect hidden layers
    # hidden_layers = []

    # start_time = time.time()

    # # Collect hidden layers per batch
    # for batch in tqdm(range(number_batches)):
    #     for i in range(per_batch):

    #         # Extract inputs from the dataset using the tokenizer
    #         inputs = {k: torch.tensor(v).to(device) for k, v in train_dataset[batch * per_batch + i].items() if k in ['input_ids', 'attention_mask']}

    #         # Perform forward pass through the model
    #         with torch.no_grad():
    #             outputs = model(**inputs, output_hidden_states=True)

    #         # Append the hidden states to the list
    #         hidden_layers.append(outputs.hidden_states)
    
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(hidden_layers[0])
    # print(f"Time taken to compute hidden states of 40 samples of {data_args.task_name} : {elapsed_time:.2f} seconds")

    ##################################################################################################################
    # Compute hidden states
    # num_data = 40
    # batch_size = 4 

    # device = torch.device("cuda")
    # model.to(device)

    # # List to store the hidden states activations
    # activations = []

    # start_time = time.time()

    # for batch_start in tqdm(range(0, num_data, batch_size)):
    #     # Extract a batch of inputs from the dataset
    #     batch_inputs = {k: torch.tensor(v).to(device) for k, v in train_dataset[batch_start:batch_start + batch_size].items() if k in ['input_ids', 'attention_mask']}

    #     # Perform forward pass through the model
    #     with torch.no_grad():
    #         outputs = model(**batch_inputs, output_hidden_states=True)

    #     # Move hidden states to CPU, convert them to numpy, and store
    #     hidden_states = [state.detach().cpu().numpy() for state in outputs.hidden_states]
    #     activations.append(np.stack(hidden_states).mean(axis=2))

    #     # Clear outputs to free memory
    #     del outputs
    #     torch.cuda.empty_cache()

    # end_time = time.time()
    # elapsed_time = end_time - start_time

    # print(f"Time taken to compute and store activations of {data_args.task_name}: {elapsed_time:.2f} seconds")



    ###########################################################################################################
    # Store the activations of hidden states for each shard
    def store_hidden_states(
        model,
        train_dataset, 
        batch_size,
        device,
        path_to_activations,
    ):
        """
        Store hidden states activations from a model into numpy arrays.

        Args:
        - model (AutoModel): The model from which to extract hidden states.
        - path_to_shards (str): Directory path where dataset shards are stored.
        - shard_id (int): ID of the dataset shard to process.
        - batch_size (int): Batch size for data loading.
        - num_workers (int): Number of worker processes for data loading.
        - device (str): Device to use for model computation.
        - path_to_activations (str): Directory path to save the activations numpy array.

        Raises:
        - FileNotFoundError: If the dataset shard file does not exist.
        """
        model.to(device)

        # Initialize the dataloader
        # dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        
        print("Size of training data of glue",data_args.task_name, len(train_dataset))

        # Store activations in a list
        print("Storing activations...")
        activations = []

        start_time = time.time()
    
        # Manually batch the data
        for i in tqdm(range(0, len(train_dataset), batch_size)):
            inputs = {k: torch.tensor(v).to(device) for k, v in train_dataset[i:i + batch_size].items() if k in ['input_ids', 'attention_mask']}
            print("Moved tensors to",device)
            with torch.no_grad():
                outputs = model(
                    **inputs, 
                    output_hidden_states=True
                )

            # Move hidden states to CPU and convert them to numpy
            hidden_states = [state.detach().cpu().numpy() for state in outputs.hidden_states]

            # Append the processed hidden states to the activations list
            activations.append(np.stack(hidden_states).mean(axis=2))

            del outputs
            torch.cuda.empty_cache()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to compute activations of {data_args.task_name} : {elapsed_time:.2f} seconds")

        os.makedirs(path_to_activations, exist_ok=True)
        # Save activations
        np.save(
            f"{path_to_activations}/activations_full_{data_args.task_name}.npy",
            np.concatenate(activations, axis=1),
        )
        print(f"Activations saved!")

        # Clean up memory
        del activations
        torch.cuda.empty_cache()
    
    device = torch.device("cuda")

    store_hidden_states(
        model=model,
        train_dataset=train_dataset,
        batch_size=4,
        device=device,
        path_to_activations="./dataset_activations"
    )

if __name__ == "__main__":
    main()