from datasets import DatasetDict, load_dataset
from collections import OrderedDict
import abc
from typing import Mapping, Optional
import numpy as np
import torch
import os
import sys 
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

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
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


# Script for custom dataset split
class AbstractTask(abc.ABC):
    name = NotImplemented
    config = NotImplemented
    prefix = NotImplemented
    split_map = None
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc",
                                         "superglue-wic", "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb",
                                         "superglue-boolq", "mnli"] 
    large_data_without_all_splits = ["qqp", "qnli", "superglue-record", "sst2"]

    split_valid_to_make_test = True
    split_train_to_make_test = False
    keep_fields_after_preprocess = ["label"]  # The fields that should be kept even after preprocessiing

    def __init__(self, config, data_args, seed=42, default_max_length=1):
        self.config = config
        self.seed = seed
        self.data_args = data_args

        self.default_max_length = default_max_length
        self.__post_init__()
    
    def __post_init__(self):
        self.split_valid_to_make_test = self.name in self.small_datasets_without_all_splits
        self.split_train_to_make_test = self.name in self.large_data_without_all_splits
    
    def load_dataset(self, split):
        tmp = load_dataset("glue",self.name,trust_remote_code=True)
        return tmp[split]

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
        return n_obs

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
           indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)


    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def preprocessor(self, example):
        return example

    def get(self, split, n_obs=None, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split in ["eval", "dev", "valid"]:
            split = "validation"
        if split_validation_test and self.split_valid_to_make_test \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_split_indices(split, dataset, validation_size=len(dataset)//2)
            dataset = self.subsample(dataset, n_obs, indices)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.split_train_to_make_test \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_split_indices(split, dataset, validation_size=1000)
            dataset = self.subsample(dataset, n_obs, indices)
        else:
            mapped_split = self.split_to_data_split[split]
            dataset = self.load_dataset(split=mapped_split)
            # shuffles the data and samples it.
            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)

        this_method = getattr(self.__class__, 'preprocessor')
        base_method = getattr(AbstractTask, 'preprocessor')
        if this_method is not base_method:
            return dataset.map(self.preprocessor)
        else:
            return dataset
        
# Prepare custom splits
class COLA(AbstractTask):
    name = "cola"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

class SST2(AbstractTask):
    name = "sst2"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

class MRPC(AbstractTask):
    name = "mrpc"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

class QQP(AbstractTask):
    name = "qqp"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

class STSB(AbstractTask):
    name = "stsb"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

class MNLI(AbstractTask):
    name = "mnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation_matched",
                           "test": "validation_matched"}

class MNLI_M(AbstractTask):
    name = "mnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation_matched",
                           "test": "validation_matched"}

class MNLI_MM(AbstractTask):
    name = "mnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_mismatched"}


class QNLI(AbstractTask):
    name = "qnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

class RTE(AbstractTask):
    name = "rte"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

class WNLI(AbstractTask):
    name = "wnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

TASK_MAPPING = OrderedDict(
    [
        ('mrpc', MRPC),
        ('cola', COLA),
        ('sst2', SST2),
        ('qnli', QNLI),
        ('rte', RTE),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('mnli-m', MNLI_M),
        ('mnli-mm', MNLI_MM),
        ('qqp', QQP),
        ('stsb', STSB),
    ]
)

class AutoTask:
    @classmethod
    def get(self, task, config, data_args, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, data_args, seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )

def main():
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, training_args = parser.parse_args_into_dataclasses()
    
    # standardize when dealing with different variations of mnli
    if "mnli" in data_args.task_name:
        data_args.task_name = "mnli"

    # Loading dataset
    if data_args.task_name is not None:
        raw_datasets = load_dataset("glue", data_args.task_name)
        task = AutoTask().get(data_args.task_name, None, None)
        raw_datasets = {
            "train": task.get("train", split_validation_test=True),
            "validation": task.get("validation", split_validation_test=True),
            "test": task.get("test", split_validation_test=True)
        }
        raw_datasets = DatasetDict(raw_datasets)
    else:
        # Loading dataset from local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files)
    
    print(raw_datasets)

    # Save the raw_datasets to disk
    raw_datasets.save_to_disk(os.path.join(training_args.output_dir, save_path, data_args.task_name))

if __name__ == "__main__":
    main()