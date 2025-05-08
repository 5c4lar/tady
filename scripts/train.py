import itertools
import pathlib
from typing import Dict, Tuple

import datasets
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
import torch.multiprocessing as multiprocessing
import tqdm
from flax import nnx
from omegaconf import DictConfig, OmegaConf
from pyarrow import parquet as pq
from tady.model.tagnn_flax import *
from torch.utils.data import DataLoader, IterableDataset

import wandb


@nnx.jit(static_argnames=["deterministic"])
def forward(model, batch, rngs=None, deterministic=True):
    output = model(batch['bytes'].astype(jnp.uint8), use_64_bit=batch['is_64'], deterministic=deterministic,
                   rngs=rngs).logits.squeeze(-1)
    return output


@nnx.jit
def loss_fn(model, batch, rngs):
    output = forward(model, batch, rngs, deterministic=False)
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


@nnx.jit
def eval_step(model, batch, metrics):
    output = forward(
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
    # split to train and test
    ds = ds.train_test_split(0.1, 0.9, seed=42)
    return ds


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(args: DictConfig):
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    
    
    
    
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
        args.connections.setting, resolve=True) # type: ignore
    num_global_connections = sum([b - a for a, b in connections.values()])
    ds_dict = load_parquets(args.dataset_dir)
    ds_dict.set_format("np")
    train_dataloader = DataLoader(ds_dict['train'], shuffle=True, collate_fn=numpy_collate, drop_last=True,
                                  batch_size=args.batch_size, pin_memory=False, persistent_workers=True, num_workers=args.process, prefetch_factor=2)
    val_dataloader = DataLoader(ds_dict['test'], shuffle=False, collate_fn=numpy_collate, drop_last=True,
                                batch_size=args.batch_size, pin_memory=False, persistent_workers=True, num_workers=args.process, prefetch_factor=2)

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
    model = FlaxLlamaForBinaryTokenClassification(
        config,
        dtype=dtype,
        rngs=rngs
    )
    # graphdef, state = nnx.split(model)
    # print(state)
    value_and_grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
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
    # tags = [args.label, args.attention, args.window_size, args.hidden_size,
    #         args.layers, config.num_global_connections, args.dtype] + ([] if args.global_connection_class else ["noclass"])
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
        for epoch in ebar:
            tbar = tqdm.tqdm(train_dataloader, position=1, leave=False)
            for i, batch in enumerate(tbar):
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
            for val_batch in val_dataloader:
                eval_step(model, val_batch, eval_metrics)
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
