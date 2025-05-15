import pathlib
from typing import Dict, Tuple

import datasets
import grain
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
import tqdm
from flax import nnx
from omegaconf import DictConfig, OmegaConf
from pyarrow import parquet as pq
from transformers import PreTrainedTokenizerFast
import wandb
from tady.model.tady_flax import *
from tady.utils.loader import chunk_data
from tady import cpp
from functools import partial
import random
@nnx.jit(static_argnames=["deterministic"])
def forward_jax(model, batch, rngs=None, deterministic=True):
    output = model(batch['bytes'].astype(jnp.uint8), is_64=batch['is_64'], deterministic=deterministic,
                   rngs=rngs).logits.squeeze(-1)
    return output


@nnx.jit(static_argnames=["deterministic"])
def forward_cpp(model, batch, rngs=None, deterministic=True):
    output = model(batch['bytes'].astype(jnp.uint8), is_64=batch['is_64'], instr_len=batch['instr_len'].astype(jnp.uint8), control_flow=batch['control_flow'].astype(jnp.int32),
                   deterministic=deterministic,
                   rngs=rngs).logits.squeeze(-1)
    return output

@nnx.jit(static_argnames=["deterministic"])
def forward_token(model, batch, rngs=None, deterministic=True):
    output = model(batch['input_ids'].astype(jnp.int32), is_64=batch['is_64'].squeeze(-1), instr_len=batch['instr_len'].astype(jnp.uint8), connections=batch['connections'].astype(jnp.int32),
                   deterministic=deterministic,
                   rngs=rngs).logits.squeeze(-1)
    return output


@nnx.jit(static_argnames=["forward_fn"])
def loss_fn(model, batch, rngs, forward_fn):
    output = forward_fn(model, batch, rngs, deterministic=False)
    loss_correct = jnp.sum(optax.sigmoid_focal_loss(
        output, batch["labels"], alpha=0.8, gamma=4.0), where=batch["mask"]
    )
    return (loss_correct, output)


class MaskedAverage(nnx.Metric):
    def __init__(self, argname: str = ''):
        self.argname = argname
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        self.total = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))

    def update(self, **kwargs):
        (mask, value) = kwargs[self.argname]
        self.total = self.total + jnp.sum(mask * value)
        self.count = self.count + jnp.sum(mask, dtype=float)

    def compute(self):
        return self.total / self.count

    def reset(self):
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        self.total = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))


@nnx.jit(static_argnames=["value_and_grad_fn"])
def train_step(model, optimizer, batch, metrics, value_and_grad_fn, rngs):
    (loss, outputs), grads = value_and_grad_fn(
        model, batch, rngs)
    optimizer.update(grads)
    pred = jax.nn.sigmoid(outputs) > 0.5
    acc = (pred == batch["labels"]).astype(jnp.float32)
    accuracy = jnp.mean(acc, where=batch["mask"])
    true_positive = pred * acc
    precision = jnp.nan_to_num(jnp.sum(
        true_positive, where=batch["mask"]) / jnp.sum(pred, where=batch["mask"]), nan=0)
    recall = jnp.nan_to_num(jnp.sum(
        true_positive, where=batch["mask"]) / jnp.sum(batch["labels"], where=batch["mask"]), nan=0)
    metrics.update(loss=loss, precision=precision,
                   recall=recall, accuracy=accuracy)


@nnx.jit(static_argnames=["forward_fn"])
def eval_step(model, batch, metrics, forward_fn):
    output = forward_fn(
        model, batch, None, deterministic=True)
    pred = jax.nn.sigmoid(output) > 0.5
    acc = (pred == batch["labels"]).astype(jnp.float32)
    accuracy = jnp.mean(acc, where=batch["mask"])
    true_positive = pred * acc
    precision = jnp.nan_to_num(jnp.sum(
        true_positive, where=batch["mask"]) / jnp.sum(pred, where=batch["mask"]), nan=0)
    recall = jnp.nan_to_num(jnp.sum(
        true_positive, where=batch["mask"]) / jnp.sum(batch["labels"], where=batch["mask"]), nan=0)
    metrics.update(precision=precision,
                   recall=recall, accuracy=accuracy)


def to_chunks(examples):
    results = {"bytes": [], "labels": [], "mask": [], "is_64": []}
    for bytes, labels, is_64 in zip(examples["bytes"], examples["labels"], examples["is_64"]):
        chunks, labels, masks = chunk_data(bytes, 8192, 64, labels)
        results["bytes"].extend(chunks.tolist())
        results["labels"].extend(labels.tolist())
        results["mask"].extend(masks.tolist())
        results["is_64"].extend([is_64] * len(chunks))
    return results


class Disassembler:
    def __init__(self):
        self.disassembler = None

    def __call__(self, example):
        if self.disassembler is None:
            self.disassembler = cpp.Disassembler()
        instr_len, _, control_flow, _ = self.disassembler.superset_disasm(
            example["bytes"], example["is_64"])
        return {
            "instr_len": instr_len,
            "control_flow": control_flow
        }

class Tokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        self.disassembler = None
        
    def overlapping_addresses(self,instr_len):
        '''
        Calculate the overlapping addresses for the instruction.

        Args:
            instr_len: A numpy array of shape [L] containing the instruction length
        Returns:
            A numpy array of shape [L, 14] containing the overlapping addresses
        '''
        offset = np.arange(0, instr_len.shape[0])
        ranges = np.arange(1, 15)
        overlapping = np.where(instr_len[:, np.newaxis] > ranges[np.newaxis, :],
                                offset[:, np.newaxis] + ranges[np.newaxis, :], -1)
        return overlapping
        
    def __call__(self, example):
        if self.disassembler is None:
            self.disassembler = cpp.Disassembler()
        instr_len, _, control_flow, _ = self.disassembler.superset_disasm(
            example["bytes"], example["is_64"])
        overlappings = self.overlapping_addresses(instr_len)
        asms = self.disassembler.disasm_to_str(example["bytes"], example["is_64"], 0)
        res = self.tokenizer(asms, max_length=16, padding="max_length", truncation=True)
        input_ids = np.array(res["input_ids"])
        token_len = np.array(res["attention_mask"]).sum(axis=-1)
        connections = np.concatenate([control_flow, overlappings], axis=-1)
        return {
            "input_ids": input_ids,
            "instr_len": token_len,
            "connections": connections,
            "is_64": example["is_64"],
            "labels": example["labels"],
            "mask": example["mask"]
        }

def numpy_collate(x):
    return {key: np.stack([i[key] for i in x]) for key in x[0]}


def load_parquets(path):
    root_dir = pathlib.Path(path)
    parquets = [i for i in root_dir.rglob("*.parquet")]
    for parquet in parquets:
        try:
            pq.ParquetFile(parquet)
        except:
            # unlink the file
            print(f"Error reading {parquet}, unlinking")
            parquet.unlink()
            continue
    parquets = [str(i) for i in root_dir.rglob("*.parquet")]
    ds = datasets.load_dataset("parquet", data_files=parquets, split="train")
    return ds


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(args: DictConfig):

    max_position_embeddings = args.model.seq_len
    attention_type = args.model.attention
    vocab_size = args.model.vocab_size
    hidden_size = args.model.hidden_size
    intermediate_size = args.model.intermediate_size
    num_attention_heads = args.model.num_attention_heads
    sliding_window = (args.model.window_size, args.model.window_size)
    num_hidden_layers = args.model.layers
    attention_dropout = args.model.dropout
    global_connection_class = args.model.global_connection_class
    token_pool = args.model.token_pool
    # print(config.num_global_connections)
    match (args.model.dtype):
        case "float32":
            dtype = jnp.float32
        case "bfloat16":
            dtype = jnp.bfloat16
        case "float16":
            dtype = jnp.float16
    connections: Dict[str, Tuple[int, int]] = OmegaConf.to_container(
        args.connections.setting, resolve=True)  # type: ignore
    num_global_connections = sum([b - a for a, b in connections.values()])
    if args.dataset_format == "parquet":
        ds = load_parquets(args.dataset_dir)
    else:
        ds = datasets.load_from_disk(args.dataset_dir)
        num_samples = args.num_samples if args.num_samples is not None else len(ds)
        ds = ds.shuffle(seed=0).take(num_samples)
        ds.set_format(type="numpy")
        ds = ds.map(to_chunks, num_proc=args.process, batched=True,
                    batch_size=1, remove_columns=ds.column_names)
        if args.model.disassembler == "cpp":
            disassembler = Disassembler()
            ds = ds.map(disassembler, num_proc=args.process)
        if args.model.disassembler == "token":
            tokenizer_fast = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer, clean_up_tokenization_spaces=False)
            tokenizer = Tokenizer(tokenizer_fast)
            ds = ds.map(tokenizer, num_proc=args.process, remove_columns=ds.column_names, writer_batch_size=100)
            ds.set_format(type="numpy")
            
    # split to train and test
    ds_dict = ds.train_test_split(0.1, 0.9, seed=42)
    ds_dict.set_format("np")
    train_ds = grain.MapDataset.source(ds_dict['train']).batch(args.batch_size)
    test_ds = grain.MapDataset.source(ds_dict['test']).batch(args.batch_size)
    train_sampler = grain.samplers.IndexSampler(
        num_records=len(train_ds),
        num_epochs=args.epoch,
        shuffle=True,
        seed=0)
    test_sampler = grain.samplers.IndexSampler(
        num_records=len(test_ds),
        num_epochs=args.epoch,
        shuffle=False,
        seed=0)
    # read_options = grain.ReadOptions(num_threads=1, prefetch_buffer_size=500)
    train_dataloader = grain.DataLoader(
        data_source=train_ds,
        sampler=train_sampler,
        # read_options=read_options,
        worker_count=4,
        worker_buffer_size=100
    )
    val_dataloader = grain.DataLoader(
        data_source=test_ds,
        sampler=test_sampler,
        # read_options=read_options,
        worker_count=4,
        worker_buffer_size=100
    )

    rngs = nnx.Rngs(params=jax.random.key(
        0), dropout=jax.random.key(1), carry=jax.random.key(2))
    num_labels = 1
    config = TAGNNConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        sliding_window=sliding_window,
        num_hidden_layers=num_hidden_layers,
        attention_type=attention_type,
        token_pool=token_pool,
        attention_dropout=attention_dropout,
        num_global_connections=num_global_connections,
        connections=connections,
        global_connection_class=global_connection_class,
        num_labels=num_labels,
        max_position_embeddings=max_position_embeddings
    )
    if args.model.disassembler == "cpp":
        model = Tady(config, dtype=dtype, rngs=rngs)
        forward_fn = forward_cpp

    elif args.model.disassembler == "jax":
        model = FlaxLlamaForBinaryTokenClassification(
            config,
            dtype=dtype,
            rngs=rngs
        )
        forward_fn = forward_jax
    elif args.model.disassembler == "token":
        model = FlaxLlamaForTokenClassification(
            config,
            dtype=dtype,
            rngs=rngs
        )
        forward_fn = forward_token
    else:
        raise ValueError(f"Invalid disassembler: {args.disassembler}")

    value_and_grad_fn = nnx.value_and_grad(
        partial(loss_fn, forward_fn=forward_fn), has_aux=True)
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1),
        optax.adamw(1e-3)
    ))
    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
        precision=nnx.metrics.Average('precision'),
        recall=nnx.metrics.Average('recall'),
        accuracy=nnx.metrics.Average('accuracy')
    )
    eval_metrics = nnx.MultiMetric(
        precision=nnx.metrics.Average('precision'),
        recall=nnx.metrics.Average('recall'),
        accuracy=nnx.metrics.Average('accuracy')
    )
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    model_id = "_".join([str(i) for i in args.tags])
    checkpoint_path = pathlib.Path(args.checkpoint) / model_id
    config_path = pathlib.Path(args.config) / model_id
    print(f"saving to {checkpoint_path.absolute()}")
    # checkpoint_steps = 100
    with wandb.init(
        project=args.wandb.project,
        config=config.to_dict(),
        tags=[str(i) for i in args.tags],
        settings=wandb.Settings(start_method="thread")
    ) as run:
        ebar = tqdm.tqdm(range(args.epoch), position=0, leave=False)
        global_step = 0
        rngs = nnx.Rngs(dropout=jax.random.key(1), carry=jax.random.key(2))
        best_eval_acc = 0
        config.save_pretrained(config_path.absolute())
        train_iter = iter(train_dataloader)
        val_iter = iter(val_dataloader)
        for epoch in ebar:
            tbar = tqdm.tqdm(range(len(train_ds)), position=1,
                             leave=False, total=len(train_ds))
            for _ in tbar:
                batch = next(train_iter)
                # for key, value in batch.items():
                #     print(key, value.shape)
                train_step(model, optimizer, batch,
                           train_metrics, value_and_grad_fn, rngs)
                metrics_dict = {f"train_{name}": f"{metric:.4f}" for name,
                                metric in train_metrics.compute().items()}
                tbar.set_postfix(metrics_dict)
                run.log({f"train_{name}": metric for name,
                        metric in train_metrics.compute().items()}, step=global_step)
                global_step += 1
                train_metrics.reset()
                # if (global_step % args.checkpoint_steps) == 0:
            for _ in range(len(test_ds)):
                val_batch = next(val_iter)
                eval_step(model, val_batch, eval_metrics, forward_fn)
                metrics_dict = {f"val_{name}": f"{metric:.4f}" for name,
                                metric in eval_metrics.compute().items()}
                ebar.set_postfix(metrics_dict)
            eval_metrics_dict = {f"val_{name}": metric for name,
                                 metric in eval_metrics.compute().items()}
            run.log(eval_metrics_dict, step=global_step)
            eval_metrics.reset()
            if (eval_metrics_dict['val_accuracy'] >= best_eval_acc):
                checkpointer.save(checkpoint_path.absolute(),
                                  nnx.state(model), force=True)


if __name__ == "__main__":
    main()
