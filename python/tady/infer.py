import pathlib
import grpc
from tady.utils.loader import load_text
from tady import cpp
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tady.utils.loader import preprocess_binary

disassembler = cpp.Disassembler()


def disassemble_batch(byte_chunks, use_64_bit):
    instr_lens = []
    control_flows = []
    for chunks, use_64 in zip(byte_chunks, use_64_bit):
        instr_len, _, control_flow, _ = disassembler.superset_disasm(
            chunks, use_64)
        instr_lens.append(instr_len)
        control_flows.append(control_flow)
    return np.array(instr_lens), np.array(control_flows)


def send_request(stub, model, byte_chunks, use_64_bit, instr_lens=None, control_flows=None):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = "serving_default"
    request.inputs["byte_sequence"].CopyFrom(
        tf.make_tensor_proto(byte_chunks)
    )
    request.inputs["use_64_bit"].CopyFrom(
        tf.make_tensor_proto(use_64_bit)
    )
    request.inputs["instr_len"].CopyFrom(
        tf.make_tensor_proto(instr_lens)
    )
    request.inputs["control_flow"].CopyFrom(
        tf.make_tensor_proto(control_flows)
    )
    result = stub.Predict(request, 100)  # 10 secs timeout
    result = result.outputs['output_0']
    result = tf.make_ndarray(result)
    result = np.array(result)
    return result


def batchify(byte_chunks: np.ndarray, masks: np.ndarray, instr_lens: np.ndarray, control_flows: np.ndarray, batch_size: int):
    # Batchify the byte chunks and masks
    batched_byte_chunks = []
    batched_masks = []
    batched_instr_lens = []
    batched_control_flows = []
    for i in range(0, len(byte_chunks), batch_size):
        batch_byte_chunks = byte_chunks[i:i + batch_size]
        batch_masks = masks[i:i + batch_size]
        batch_instr_lens = instr_lens[i:i + batch_size]
        batch_control_flows = control_flows[i:i + batch_size]
        if len(batch_byte_chunks) < batch_size:
            # Pad the batch with zeros
            pad_size = batch_size - len(batch_byte_chunks)
            batch_byte_chunks = np.pad(batch_byte_chunks, ((
                0, pad_size), (0, 0)), mode='constant', constant_values=0x90)
            batch_masks = np.pad(
                batch_masks, ((0, pad_size), (0, 0)), mode='constant', constant_values=False)
            batch_instr_lens = np.pad(
                batch_instr_lens, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            batch_control_flows = np.pad(batch_control_flows, ((
                0, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)
        batched_byte_chunks.append(np.array(batch_byte_chunks, dtype=np.uint8))
        batched_masks.append(np.array(batch_masks, dtype=np.bool))
        batched_instr_lens.append(np.array(batch_instr_lens, dtype=np.uint8))
        batched_control_flows.append(
            np.array(batch_control_flows, dtype=np.int32))
    return batched_byte_chunks, batched_masks, batched_instr_lens, batched_control_flows


def process_file(args, path, model, stub, disassembler):
    byte_chunks, masks, use_64_bit, base_addr = preprocess_binary(
        path, args.seq_len, section_name=args.section_name)
    batched_instr_lens = []
    batched_control_flows = []
    for sequence in byte_chunks:
        instr_lens, _, control_flows, _ = disassembler.superset_disasm(
            sequence, use_64_bit)
        batched_instr_lens.append(instr_lens)
        batched_control_flows.append(control_flows)
    batched_instr_lens = np.array(batched_instr_lens, dtype=np.uint8)
    batched_control_flows = np.array(batched_control_flows, dtype=np.int32)
    batched_byte_chunks, batched_masks, batched_instr_lens, batched_control_flows = batchify(
        byte_chunks, masks, batched_instr_lens, batched_control_flows, args.batch_size)
    logits = []
    for sequence, mask, instr_lens, control_flows in zip(batched_byte_chunks, batched_masks, batched_instr_lens, batched_control_flows):
        is_64_bit = np.array([use_64_bit] * len(sequence), dtype=np.bool)
        result = send_request(stub, model, sequence,
                              is_64_bit, instr_lens, control_flows)
        logits.append(result[mask])
    logits = np.concatenate(logits, axis=0)
    pred = logits > 0
    result = {
        "logits": logits,
        "pred": pred,
        "base_addr": np.array(base_addr, dtype=np.uint64),
    }
    # print(result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to the binary file")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--section_name", type=str, help="Section name")
    parser.add_argument("--output_path", type=str, help="Output path")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host of the model server.")
    parser.add_argument("--port", type=int, default=8500,
                        help="Port of the model server.")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int,
                        default=8192, help="Sequence length")
    args = parser.parse_args()
    options = [
        # Set to slightly larger than the problematic message
        ('grpc.max_receive_message_length', 16777327 + 1024),
        # You can also set a much larger limit if you expect even bigger messages, e.g., 50 * 1024 * 1024 for 50MB
        # ('grpc.max_receive_message_length', 50 * 1024 * 1024),
    ]
    channel = grpc.insecure_channel(
        f"{args.host}:{args.port}", options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result = process_file(args, args.path, args.model, stub, disassembler)
    pathlib.Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_path, **result)


if __name__ == "__main__":
    main()
