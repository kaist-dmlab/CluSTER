from dataclasses import dataclass, field
from typing import cast
from typing import Dict, Optional, Sequence, List

import transformers
import torch
from datasets import load_dataset
from transformers import HfArgumentParser, Trainer, TrainingArguments

from CluSTER.llm_wrapper import (
    DecodingConfig,
    EncodingConfig,
    TokenizationContext,
    get_model_context,
    pad_sequences,
)
from CluSTER.prompt_template import MAGICODER_PROMPT
from CluSTER.utils import N_CORES
import random
import numpy as np
from CluSTER.coreset_trainer import CustomTrainer
from CluSTER.call_back import SamplerEpochSetterCallback, PruneAndClusteringinwithMeanCallback, UniformCallback

import time

@dataclass(frozen=True)
class ModelArguments:
    model_key: str
    model_name_or_path: str | None = None


# Ignored index in CrossEntropyLoss
IGNORED_INDEX = -100


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def map_dataset(
    examples: dict[str, list[str]],
    args: "Args",
    context: TokenizationContext,
) -> dict:
    instructions = examples["instruction"]
    responses = examples["response"]

    prompts = [
        MAGICODER_PROMPT.format(instruction=instruction, response="")
        for instruction in instructions
    ]
    completions = responses

    assert len(prompts) == len(completions)
    prompt_config = EncodingConfig(add_bos=True, add_eos=False)
    completion_config = EncodingConfig(add_bos=False, add_eos=True)
    prompt_id_batches = context.encode(prompt_config, prompts)
    completion_id_batches = context.encode(completion_config, completions)
    # prompt_id_batches = context.tokenization_context.encode(prompt_config, prompts)
    # completion_id_batches = context.tokenization_context.encode(
    #     completion_config, completions
    # )
    assert len(prompt_id_batches) == len(completion_id_batches)
    untruncated_input_ids = [
        (instruction_ids + response_ids)
        for instruction_ids, response_ids in zip(
            prompt_id_batches, completion_id_batches
        )
    ]
    exceeding_length = [
        len(input_id) > args.max_training_seq_length
        for input_id in untruncated_input_ids
    ]
    input_ids = [
        input_id[: args.max_training_seq_length] for input_id in untruncated_input_ids
    ]
    # NOTE: no need to set EOF to IGNORED_INDEX as it is *implicitly* ignored inside
    # the model.forward that shifts the logits left by 1
    labels = [
        (list(map(lambda _: IGNORED_INDEX, instruction_ids)) + response_ids)[
            : args.max_training_seq_length
        ]
        for instruction_ids, response_ids in zip(
            prompt_id_batches, completion_id_batches
        )
    ]
    # `len` of each returned value must be the same, which is required by `tokenizer.map`
    # After `map`, they are treated as individual pieces of data, not as a batch.
    assert len(input_ids) == len(labels)
    for input_id_batch, label_batch in zip(input_ids, labels):
        assert len(input_id_batch) == len(label_batch)
    print(context.decode(DecodingConfig.default(), input_ids[0:])[0])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "exceeding_length": exceeding_length,
    }


def get_data_collator(args: "Args", pad_token_id: int):
    """Pad input_ids to the right, create labels by setting the padding tokens to -100, and
    create attention_mask to ignore the padding tokens"""

    def collate(examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids_unpadded = [example["input_ids"] for example in examples]
        labels_unpadded = [example["labels"] for example in examples]
        padding_length = (
            args.max_training_seq_length if args.pad_to_max_length else None
        )
        input_ids = pad_sequences(
            input_ids_unpadded, pad_token_id, "right", padding_length=padding_length
        )
        labels = pad_sequences(
            labels_unpadded, IGNORED_INDEX, "right", padding_length=padding_length
        )

        assert input_ids.shape == labels.shape
        assert len(input_ids) == len(examples)
        # Enforced in `map_raw_dataset`
        assert input_ids.shape[-1] <= args.max_training_seq_length
        if args.pad_to_max_length:
            assert input_ids.shape[-1] == args.max_training_seq_length

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_token_id),
        }

    return collate

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adafactor")
    prune: Optional[str] = field(default="close", metadata={"help": "Type of the Pruner to use."})
    sampling_type: Optional[str] = field(default="interleaved", metadata={"help": "Type of the Sampler to use."})
    weight: bool = field(default=True, metadata={"help": "Whether to use weights in coreset selection."})
    cluster_sizes: List[int] = field(
        default_factory=lambda: [1,1,1,1],
        metadata={"help": "Integers for cluster sizes."}
    )
    seed: Optional[int] = field(default = 42, metadata={"help": "seed"})
    ratio: Optional[int] = field(default = 100, metadata={"help": "ratio for coreset selection"})
    badge_batch: Optional[int] = field(default = 4, metadata={"help": "batch size for badge"})
    badge_forward_chunk_mult: Optional[int] = field(
        default=1,
        metadata={"help": "Multiplier for per-forward BADGE micro-chunk size (higher=faster, more memory)."},
    )
    badge_cleanup_interval: Optional[int] = field(
        default=0,
        metadata={"help": "Run expensive CUDA cache cleanup every N BADGE sub-batches (0 disables)."},
    )
    weight_mode: Optional[str] = field(default="inv")

@dataclass(frozen=True)
class Args:
    datafile_paths: list[str] = field(default_factory=list)
    max_training_seq_length: int = field(default=1216)
    pad_to_max_length: bool = field(default=False)
    eval_dataset_size: float = field(
        default=0.05, metadata={"help": "0--1 means ratio, >1 means number of examples"}
    )
    use_flash_attention: bool = field(default=False)
    uniform: bool = field(default=False, metadata={"help": "Whether to use uniform sampling instead of coreset selection."})


def train():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, Args))
    model_args, training_args, args = cast(
        tuple[ModelArguments, TrainingArguments, Args],
        parser.parse_args_into_dataclasses(),
    )
    seed_everything(training_args.seed)
    dataset = load_dataset("json", data_files=args.datafile_paths, split="train")

    model_key = model_args.model_key
    if (model_name_or_path := model_args.model_name_or_path) is None:
        model_name_or_path = model_key

    tokenization_context = TokenizationContext.from_model_key(
        model_key, model_name_or_path
    )
    # if dataset_config.dpo_jsonl_path is None or dataset_config.dpo_sft:
    train_dataset = dataset.map(
        function=map_dataset,
        fn_kwargs=dict(args=args, context=tokenization_context),
        batched=True,
        num_proc=N_CORES,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,  # not args.overwrite_cache
        desc="Running tokenizer on train dataset",
    )
    msg = f"#Examples truncated: {sum(train_dataset['exceeding_length'])} / {len(train_dataset)}"
    print(msg)
    # else:
    #     train_dataset = dataset

    # Shuffling
    if training_args.eval_steps is None and training_args.eval_strategy == "no": ##0219 edit
        train_dataset = train_dataset.shuffle(seed=training_args.seed)
        eval_dataset = None
    else:
        print("Splitting dataset")
        split_dataset = train_dataset.train_test_split(
            test_size=args.eval_dataset_size,
            shuffle=True,
            seed=training_args.seed,
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    state = get_model_context(
        model_key,
        model_name_or_path,
        tokenization_context,
        inference_mode=False,
        use_flash_attention=args.use_flash_attention,
    )

    print("Parallel mode:", training_args.parallel_mode)
    data_collator = get_data_collator(args, state.tokenization_context.pad_token_id)

    # neftune_noise_alpha
    trainer = CustomTrainer(
        model=state.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=state.tokenization_context.tokenizer,
        # eval_dataset=small_eval_dataset,
        # compute_metrics=compute_metrics,
    )
    print("Using precomputed gradient-based ordering before trainer.train()")
    trainer.add_callback(SamplerEpochSetterCallback(trainer))

    if args.uniform:
        print("Building UniformCallback for one-time precompute")
        precompute_callback = UniformCallback(
            trainer,
            dataset=trainer.train_dataset,
            tokenizer=tokenization_context.tokenizer,
        )
    else:
        print("Building PruneAndClusteringinwithMeanCallback for one-time precompute")
        precompute_callback = PruneAndClusteringinwithMeanCallback(
            trainer,
            dataset=trainer.train_dataset,
            tokenizer=tokenization_context.tokenizer,
        )

    start = time.time()
    # Ensure trainer has initialized sampler object referenced by the callback.
    trainer.get_train_dataloader()

    # Trigger the callback pipeline once before training starts.
    if trainer.state.epoch is None:
        trainer.state.epoch = 0.0
    start1 = time.time()
    precompute_callback.on_train_begin(
        training_args,
        trainer.state,
        trainer.control,
        model=trainer.model,
    )
    end = time.time()
    print("Finished one-time BADGE precompute and sampler ordering in time: ", end-start)

    print("From dataloader to one-time BADGE precompute and sampler ordering in time: ", end-start1)

    # NOTE: the checkpoint will override the initialized model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    state.tokenization_context.tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
