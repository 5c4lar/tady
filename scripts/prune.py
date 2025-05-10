from tady.utils.loader import load_text
from tady.model.disasm_jax import DisasmJax
# from tady.graph import cpp
import jax
import numpy as np
import jax.numpy as jnp

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prune the graph')
    parser.add_argument('--file', type=str, required=True, help='Path to the binary file')
    args = parser.parse_args()
    file_path = args.file
    # Load the binary file

    
    text_array, use_64_bit, base_addr = load_text(file_path)
        
    # print(text_array[:100])
    print(hex(base_addr))
    print(use_64_bit)
    # cpp_disasm = cpp.ldasm(text_array, use_64_bit)
    # print(cpp_disasm[:10])
    disassembler = DisasmJax()
    text_array = jax.device_put(text_array)
    jax_disasm, opcode_modrm = disassembler(text_array, np.array(use_64_bit, dtype=bool))
    print(jax_disasm[:10], opcode_modrm[:10])
    
    # print(jax_disasm[:10])
    # diff = jax_disasm != cpp_disasm
    # # get the rows where the values are different
    # different_rows = np.where(diff.any(axis=1))[0]
    # print(f"Different rows: {len(different_rows)} {different_rows}")
    # print(different_rows[:10])
    # print(jax_disasm[different_rows][:10])
    # print(cpp_disasm[different_rows][:10])
    # jax_disasm = disassembler.disasm_seq(text_array, np.array(use_64_bit, dtype=bool))[:, :11]
    # jax_disasm = jax_disasm_sequence(text_array, np.array(use_64_bit, dtype=bool))[:,:11]
    # cpp_disasm = jax_disasm_sequence(text_array, np.array(use_64_bit, dtype=bool))[:,:11]
    # jax_disasm = cpp.ldasm(text_array, use_64_bit)
    # text_array, use_64_bit, base_addr = load_text(file_path)
    
    # diff = jax_disasm != cpp_disasm
    # # get the rows where the values are different
    # diff_rows = np.where(diff.any(axis=1))[0]
    # # print the rows where the values are different
    # print(f"Different rows: {len(diff_rows)} {diff_rows}")
    # print(jax_disasm[diff_rows][:10])
    # print(cpp_disasm[diff_rows][:10])
    # print(text_array.shape)
    # instruction, control_flow, instr_len = preprocess_binary(text_array, np.array(use_64_bit, dtype=bool))
    # edges = jax.device_get(control_flow[:, :2])
    # weights = np.ones(edges.shape[0], dtype=np.float32)
    # cf = np.ones(edges.shape[0], dtype=bool)
    # print(edges)
    # wccs = cpp.process_graph_pipeline(edges, weights, cf)
    # print(f"Number of WCCs: {wccs}")