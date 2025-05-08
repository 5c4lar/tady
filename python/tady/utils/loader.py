import lief
import numpy as np

def chunk_data(byte_sequence, seq_len, window_size, gt=None):
    # Turn the byte sequence into chunks of seq_len with window_size overlap
    byte_chunks = []
    labels = []
    masks = []
    chunk_step = seq_len - 2 * window_size
    labels_array = gt if gt is not None else np.zeros(len(byte_sequence), dtype=bool)
    for i in range(0, len(byte_sequence), chunk_step):
        start = i
        end = i + seq_len
        current_chunk = np.frombuffer(byte_sequence[start:end], dtype=np.uint8)
        current_labels = labels_array[start:end]
        mask = np.ones(seq_len, dtype=bool)
        real_len = len(current_chunk)
        if real_len < seq_len:
            current_chunk = np.pad(current_chunk, (0, seq_len - real_len), constant_values=0)
            current_labels = np.pad(current_labels, (0, seq_len - real_len), constant_values=False)
            mask[real_len:] = False
        mask[-window_size:] = False
        if i != 0:
            mask[:window_size] = False
        byte_chunks.append(current_chunk.tolist())
        labels.append(current_labels.tolist())
        masks.append(mask.tolist())
    if gt is not None:
        return byte_chunks, labels, masks
    else:
        return byte_chunks, masks

def preprocess_binary(file_path, seq_len=8192, window_size=64):
    # Load the binary file
    binary = lief.parse(str(file_path))
    text_section = binary.get_section(".text")
    base_addr = text_section.virtual_address
    text_array = np.frombuffer(text_section.content, dtype=np.uint8)
    byte_chunks, masks = chunk_data(text_array, seq_len, window_size)
    use_64_bit = binary.header.machine_type == lief.ELF.ARCH.X86_64
    return byte_chunks, masks, use_64_bit, base_addr

if __name__ == "__main__":
    file_path = "/bin/bash"
    byte_chunks, masks, use_64_bit, base_addr = preprocess_binary(file_path)
    print(f"Base address: {base_addr}")
    print(f"Use 64 bit: {use_64_bit}")
    byte_chunks = np.array(byte_chunks)
    masks = np.array(masks)
    print(f"Byte chunks: {byte_chunks}")
    print(f"Masks: {masks}")

