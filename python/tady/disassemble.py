import argparse
import pathlib
from types import SimpleNamespace
from typing import Optional, Tuple, Dict, Any

import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from tady import cpp
from tady.infer import batchify, disassembler
from tady.prune import get_pdt
from tady.utils.loader import preprocess_binary, load_text


def _send_request(
    stub: prediction_service_pb2_grpc.PredictionServiceStub,
    model: str,
    byte_chunks: np.ndarray,
    use_64_bit: np.ndarray,
    instr_lens: np.ndarray,
    control_flows: np.ndarray,
) -> np.ndarray:
    """
    Thin wrapper around TF Serving predict API.
    """
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
    result = stub.Predict(request, 100)
    result = result.outputs["output_0"]
    result = tf.make_ndarray(result)
    return np.array(result)


def run_model_on_binary(
    path: pathlib.Path,
    model: str,
    host: str = "localhost",
    port: int = 8500,
    section_name: Optional[str] = None,
    batch_size: int = 32,
    seq_len: int = 8192,
) -> Dict[str, Any]:
    """
    Run the neural model on a binary and return logits and raw predictions.

    This reuses the same preprocessing and batching logic as `tady.infer`,
    but is structured as a reusable function.
    """
    byte_chunks, masks, use_64_bit, base_addr = preprocess_binary(
        path, seq_len=seq_len, section_name=section_name
    )

    # Per-sequence disassembly for model features
    batched_instr_lens = []
    batched_control_flows = []
    for sequence in byte_chunks:
        instr_lens, _, control_flows, _ = disassembler.superset_disasm(
            sequence, use_64_bit
        )
        batched_instr_lens.append(instr_lens)
        batched_control_flows.append(control_flows)

    batched_instr_lens = np.array(batched_instr_lens, dtype=np.uint8)
    batched_control_flows = np.array(batched_control_flows, dtype=np.int32)

    (
        batched_byte_chunks,
        batched_masks,
        batched_instr_lens,
        batched_control_flows,
    ) = batchify(
        np.array(byte_chunks),
        np.array(masks),
        batched_instr_lens,
        batched_control_flows,
        batch_size,
    )

    options = [
        ("grpc.max_receive_message_length", 16777327 + 1024),
    ]
    channel = grpc.insecure_channel(f"{host}:{port}", options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    logits_list = []
    for sequence, mask, instr_lens, control_flows in zip(
        batched_byte_chunks,
        batched_masks,
        batched_instr_lens,
        batched_control_flows,
    ):
        is_64_bit = np.array([use_64_bit] * len(sequence), dtype=np.bool_)
        result = _send_request(
            stub,
            model,
            sequence,
            is_64_bit,
            instr_lens,
            control_flows,
        )
        logits_list.append(result[mask])

    logits = np.concatenate(logits_list, axis=0)
    pred = logits > 0
    return {
        "logits": logits,
        "pred": pred,
        "base_addr": np.array(base_addr, dtype=np.uint64),
    }


def disassemble(
    path: pathlib.Path,
    model: str,
    host: str = "localhost",
    port: int = 8500,
    section_name: Optional[str] = None,
    batch_size: int = 32,
    seq_len: int = 8192,
    use_pdt: bool = False,
    pdt_cache_path: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """
    Disassemble a binary using the neural model, with optional PDT pruning.

    Returns a dict containing at least:
      - logits: raw model scores per byte
      - pred: raw boolean predictions per byte
      - base_addr: base virtual address of the section
      - pred_pruned: (if use_pdt) predictions after PDT pruning
    """
    path = pathlib.Path(path)

    # Run the model to get logits and raw predictions
    result = run_model_on_binary(
        path=path,
        model=model,
        host=host,
        port=port,
        section_name=section_name,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    if not use_pdt:
        return result

    # Build PDT on the full text section and prune predictions
    text_array, use_64_bit, base_addr = load_text(
        path, section_name=section_name
    )
    if text_array.shape[0] != result["logits"].shape[0]:
        raise ValueError(
            "Length mismatch between text_array and logits; "
            "make sure seq_len and section_name are consistent."
        )

    dis = cpp.Disassembler()
    _, flow_kind, _, successors = dis.superset_disasm(text_array, use_64_bit)
    cf = flow_kind > 1

    pdt = get_pdt(successors, cf, pdt_cache_path)
    pruned_indices = pdt.prune(result["logits"])

    pruned_pred = np.zeros_like(result["pred"], dtype=np.bool_)
    pruned_pred[pruned_indices] = True

    result["pred_pruned"] = pruned_pred
    # For convenience, also expose the base address as uint64 scalar
    result["base_addr"] = np.array(base_addr, dtype=np.uint64)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Neural disassembly helper that calls the Tady model and "
            "optionally applies PDT-based pruning."
        )
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the binary file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name as configured in tensorflow-serving.",
    )
    parser.add_argument(
        "--section_name",
        type=str,
        default=None,
        help="Section name to disassemble (default: implementation-defined).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="If set, save results to this NPZ file.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host of the model server (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8500,
        help="Port of the model server (default: 8500).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=8192,
        help="Sequence length for preprocessing.",
    )
    parser.add_argument(
        "--use_pdt",
        action="store_true",
        help="If set, apply PDT-based pruning to the model predictions.",
    )
    parser.add_argument(
        "--pdt_cache",
        type=str,
        default=None,
        help="Optional path to cache PDT data (wccs, dom_tree).",
    )

    args = parser.parse_args()
    pdt_cache_path = (
        pathlib.Path(args.pdt_cache) if args.pdt_cache is not None else None
    )

    result = disassemble(
        path=pathlib.Path(args.path),
        model=args.model,
        host=args.host,
        port=args.port,
        section_name=args.section_name,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        use_pdt=args.use_pdt,
        pdt_cache_path=pdt_cache_path,
    )

    if args.output_path is not None:
        out_path = pathlib.Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **result)
    else:
        # Minimal textual summary if no output file is requested
        base_addr = int(result["base_addr"])
        pred = result["pred_pruned"] if "pred_pruned" in result else result["pred"]
        addrs = np.where(pred)[0].astype(np.uint64) + base_addr
        print([hex(int(a)) for a in addrs])


if __name__ == "__main__":
    main()
