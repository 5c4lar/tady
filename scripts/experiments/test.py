import json
import pathlib
from multiprocessing.pool import ThreadPool as Pool
import grpc
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tady.utils.loader import preprocess_binary, len_to_overlappings
import tensorflow as tf
from transformers import PreTrainedTokenizerFast
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from typing import List
from tady import cpp

import os
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

def disassemble_batch(byte_chunks, use_64_bit):
    disassembler = cpp.Disassembler()
    instr_lens = []
    control_flows = []
    for chunks, use_64 in zip(byte_chunks, use_64_bit):
        instr_len, _, control_flow, _ = disassembler.superset_disasm(
            chunks, use_64)
        instr_lens.append(instr_len)
        control_flows.append(control_flow)
    return np.array(instr_lens), np.array(control_flows)

def tokenize_batch(byte_chunks, use_64_bit, tokenizer):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer, clean_up_tokenization_spaces=False)
    disassembler = cpp.Disassembler()
    instr_lens = []
    connections = []
    input_ids = []
    for chunks, use_64 in zip(byte_chunks, use_64_bit):
        instr_len, _, control_flow, _ = disassembler.superset_disasm(chunks, use_64)
        overlappings = len_to_overlappings(instr_len)
        asms = disassembler.disasm_to_str(chunks, use_64, 0)
        res = tokenizer(asms, max_length=16, padding="max_length", truncation=True)
        input_ids.append(np.array(res["input_ids"], dtype=np.int32))
        instr_lens.append(np.array(res["attention_mask"]).sum(axis=-1, dtype=np.uint8))
        connections.append(np.concatenate([control_flow, overlappings], axis=-1, dtype=np.int32))
    return np.array(input_ids), np.array(instr_lens), np.array(connections)

def send_request(stub, model, byte_chunks, use_64_bit, disassembler, tokenizer=None):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = "serving_default"
    if disassembler == "jax":
        request.inputs["byte_sequence"].CopyFrom(
            tf.make_tensor_proto(byte_chunks)
        )
        request.inputs["use_64_bit"].CopyFrom(
            tf.make_tensor_proto(use_64_bit)
        )
    elif disassembler == "cpp":
        request.inputs["byte_sequence"].CopyFrom(
            tf.make_tensor_proto(byte_chunks)
        )
        request.inputs["use_64_bit"].CopyFrom(
            tf.make_tensor_proto(use_64_bit)
        )
        instr_lens, control_flows = disassemble_batch(byte_chunks, use_64_bit)
        request.inputs["instr_len"].CopyFrom(
            tf.make_tensor_proto(instr_lens)
        )
        request.inputs["control_flow"].CopyFrom(
            tf.make_tensor_proto(control_flows)
        )
    elif disassembler == "token":
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tokenize_batch, byte_chunks, use_64_bit, tokenizer)
            input_ids, instr_lens, connections = future.result()
        request.inputs["input_ids"].CopyFrom(
            tf.make_tensor_proto(input_ids)
        )
        request.inputs["instr_len"].CopyFrom(
            tf.make_tensor_proto(instr_lens)
        )
        request.inputs["connections"].CopyFrom(
            tf.make_tensor_proto(connections)
        )
        request.inputs["use_64_bit"].CopyFrom(
            tf.make_tensor_proto(use_64_bit)
        )
    result = stub.Predict(request, 100)  # 10 secs timeout
    # Transform the result to a numpy array
    # Extract the first output tensor from the outputs map
    # Convert the result to a numpy array
    result = result.outputs['output_0']
    result = tf.make_ndarray(result)
    result = np.array(result)
    return result


def batchify(byte_chunks: np.ndarray, masks: np.ndarray, batch_size: int):
    # Batchify the byte chunks and masks
    batched_byte_chunks = []
    batched_masks = []
    for i in range(0, len(byte_chunks), batch_size):
        batch_byte_chunks = byte_chunks[i:i + batch_size]
        batch_masks = masks[i:i + batch_size]
        if len(batch_byte_chunks) < batch_size:
            # Pad the batch with zeros
            pad_size = batch_size - len(batch_byte_chunks)
            batch_byte_chunks = np.pad(batch_byte_chunks, ((0, pad_size), (0, 0)), mode='constant', constant_values=0x90)
            batch_masks = np.pad(batch_masks, ((0, pad_size), (0, 0)), mode='constant', constant_values=False)
        batched_byte_chunks.append(np.array(batch_byte_chunks, dtype=np.uint8))
        batched_masks.append(np.array(batch_masks, dtype=np.bool))
    return batched_byte_chunks, batched_masks

def process_file(arg):
    args, file, output, model, stub = arg
    if file.is_file():
        rel_path = file.relative_to(args.dir)
        output_path = pathlib.Path(output) / (str(rel_path) + ".npz")
        if output_path.exists():
            return
        byte_chunks, masks, use_64_bit, base_addr = preprocess_binary(file)
        batched_byte_chunks, batched_masks = batchify(
            byte_chunks, masks, args.batch_size)
        logits = []
        for sequence, mask in zip(batched_byte_chunks, batched_masks):
            is_64_bit = np.array([use_64_bit] * len(sequence), dtype=np.bool)
            result = send_request(stub, model, sequence, is_64_bit, args.model.disassembler, args.tokenizer)
            logits.append(result[mask])
        logits = np.concatenate(logits, axis=0).flatten()
        pred = logits > args.threshold
        result = {
            "scores": logits,
            "pred": pred,
            "base_addr": np.array(base_addr, dtype=np.uint64),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)


@hydra.main(version_base=None, config_path="conf", config_name="test")
def main(args: DictConfig):
    model_id = "_".join([str(i) for i in args.tags])
    root_dir = pathlib.Path(args.dir)
    output = pathlib.Path(args.output) / model_id
    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    model = args.model_id if args.model_id else model_id
    print(f"Testing {model} on {root_dir}")
    files = [i for i in root_dir.rglob("*") if i.is_file()]
    if args.num_samples:
        import random
        random.seed(0)
        files = random.sample(files, args.num_samples)
    with Pool(args.process) as pool, tqdm(total=len(files)) as pbar:
        for result in pool.imap_unordered(process_file, [(args, file, output, model, stub) for file in files]):
            pbar.update()
            pbar.refresh()


if __name__ == "__main__":
    main()
