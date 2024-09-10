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
    DataCollatorWithPadding,
    AutoTokenizer, 
    PretrainedConfig,
    AutoModelForSequenceClassification,
    EvalPrediction,
    default_data_collator,
)
import torch 
from tqdm import tqdm
import time
import numpy as np
from peft import LoraConfig
from trl import SFTTrainer
from accelerate import PartialState

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
    
    device_string = PartialState().process_index

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
        device_map={'':device_string}, 
        torch_dtype=torch.bfloat16
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

        if label_to_id is not None and "label" in examples:
            result["labels"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        else:
            result["labels"] = examples["label"]
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

    # Get the metric function
    from datasets import load_metric
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name, trust_remote_code=True)
    else:
        metric = load_metric("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
    
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    import pickle

    # Load the rank pattern from the pickle file
    rank_pattern_file_path = f'./rank_pattern/rank_pattern_{data_args.task_name}.pkl'
    with open(rank_pattern_file_path, 'rb') as file:
        rank_pattern = pickle.load(file)

    print("Loaded Rank Pattern:", rank_pattern)

    # Load the alpha pattern from the pickle file
    alpha_pattern_file_path = f'./alpha_pattern/alpha_pattern_{data_args.task_name}.pkl'
    with open(alpha_pattern_file_path, 'rb') as file:
        alpha_pattern = pickle.load(file)

    print("Loaded alpha Pattern:", alpha_pattern)

    mean_rank_path = f'./mean_rank/mean_rank_{data_args.task_name}.pkl'
    with open(mean_rank_path, 'rb') as file:
        mean_rank = pickle.load(file)
    mean_rank = int(mean_rank)
    print("Loaded mean rank:", mean_rank)

    print("**************************************")
    print(model)
    # Set training arguments
    # rank = 1
    training_arguments = TrainingArguments(
        output_dir = "./results",
        report_to="none",
        num_train_epochs = 10,
        fp16 = False,
        bf16 = False,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 1,
        gradient_checkpointing = True,
        max_grad_norm = 0.3,
        learning_rate = 1e-4,
        weight_decay = 0.001,
        optim = "paged_adamw_32bit",
        lr_scheduler_type = "cosine",
        max_steps = -1,
        warmup_ratio = 0.03,
        group_by_length = True,
        save_steps = 0,
        logging_steps = 200,
        label_names=["labels"],
        eval_steps=200,  # Evaluate every 200 steps
        evaluation_strategy="steps",  
        ddp_find_unused_parameters=False,
    )
    # LoRA configuration
    # alpha/rank = 8
    peft_config = LoraConfig(
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj','k_proj','v_proj','o_proj'],
        rank_pattern = rank_pattern,
        alpha_pattern = alpha_pattern,
        # r = mean_rank,
        # lora_alpha = 32
    )
    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset = val_dataset,
        peft_config=peft_config,
        # dataset_text_field="text",
        compute_metrics=compute_metrics,
        max_seq_length= data_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator = data_collator,
    )
    metrics = trainer.train()
    print("train", metrics)

     # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    eval_datasets = [val_dataset]

    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        print("eval", metrics)

    tasks = [data_args.task_name]
    predict_datasets = [test_dataset]

    for predict_dataset, task in zip(predict_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=predict_dataset)
        print("test", metrics)

if __name__ == "__main__":
    main()