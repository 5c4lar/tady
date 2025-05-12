import lief
import numpy as np

def chunk_data(byte_sequence, seq_len, window_size, gt=None):
    """
    Turns a byte sequence into chunks of seq_len with window_size overlap.

    Args:
        byte_sequence (np.ndarray): The input byte sequence.
        seq_len (int): The length of each chunk.
        window_size (int): The size of the overlap window on each side of a chunk.
        gt (array-like, optional): Ground truth labels, same length as byte_sequence.

    Returns:
        tuple: Depending on gt:
            - (byte_chunks_arr, labels_arr, masks_arr) if gt is provided.
            - (byte_chunks_arr, masks_arr) if gt is None.
            All returned arrays are NumPy arrays.
    """
    if not isinstance(byte_sequence, np.ndarray):
        raise TypeError("byte_sequence must be a numpy array.")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if window_size < 0:
        raise ValueError("window_size must be non-negative.")

    chunk_step = seq_len - 2 * window_size
    if chunk_step <= 0:
        raise ValueError(
            "seq_len must be greater than 2 * window_size for chunk_step to be positive."
        )

    n_bytes = len(byte_sequence)
    if n_bytes == 0:
        num_chunks = 0
    else:
        # Calculate the number of chunks based on the starting positions
        num_chunks = (n_bytes - 1) // chunk_step + 1

    # Initialize output arrays
    byte_chunks_arr = np.zeros((num_chunks, seq_len), dtype=np.uint8)
    masks_arr = np.zeros((num_chunks, seq_len), dtype=bool) # Will be filled carefully

    # Prepare labels_array and labels_arr regardless of gt, to match original logic's internal processing.
    # labels_arr is only returned if gt is not None.
    if gt is not None:
        if not hasattr(gt, '__len__') or len(gt) != n_bytes:
            raise ValueError("gt (ground truth) must have the same length as byte_sequence.")
        labels_array = np.asarray(gt, dtype=bool)
    else:
        labels_array = np.zeros(n_bytes, dtype=bool)
    
    # This array will be populated even if gt is None, but only returned if gt is not None.
    labels_arr = np.zeros((num_chunks, seq_len), dtype=bool)

    current_pos_in_sequence = 0
    for idx in range(num_chunks):
        start = current_pos_in_sequence
        end = start + seq_len

        # Process byte chunk
        actual_chunk_bytes = byte_sequence[start:end]
        real_len = len(actual_chunk_bytes)

        if real_len > 0:
            byte_chunks_arr[idx, :real_len] = actual_chunk_bytes
            # The rest of byte_chunks_arr[idx] remains 0 (padding)

        # Process labels
        if real_len > 0:
            labels_data_for_chunk = labels_array[start : start + real_len]
            labels_arr[idx, :real_len] = labels_data_for_chunk
            # The rest of labels_arr[idx] remains False (padding)

        # Process mask
        # Start with mask as all True for seq_len, then mark invalid parts.
        mask = np.ones(seq_len, dtype=bool)
        if real_len < seq_len:
            mask[real_len:] = False  # Mark padding area as invalid

        # Mark window overlaps as invalid
        # This logic should handle window_size == 0 correctly (no-op for masks)
        # and window_size > seq_len (masks entire chunk).
        if window_size > 0:
            # Mask out the end window (right side)
            # Equivalent to mask[-window_size:] = False but robust for small seq_len
            mask[max(0, seq_len - window_size):seq_len] = False
            
            if idx != 0:  # Not the first chunk
                # Mask out the start window (left side)
                # Equivalent to mask[:window_size] = False but robust
                mask[:min(window_size, seq_len)] = False
        
        masks_arr[idx] = mask
        
        current_pos_in_sequence += chunk_step

    if gt is not None:
        return byte_chunks_arr, labels_arr, masks_arr
    else:
        return byte_chunks_arr, masks_arr
    
def load_text(file_path):
    binary = lief.parse(str(file_path))
    text_section = binary.get_section(".text")
    base_addr = text_section.virtual_address
    text_bytes = bytes(text_section.content)
    text_array = np.frombuffer(text_bytes, dtype=np.uint8)
    use_64_bit = binary.header.machine_type == lief.ELF.ARCH.X86_64
    return text_array, use_64_bit, base_addr

def preprocess_binary(file_path, seq_len=8192, window_size=64):
    text_array, use_64_bit, base_addr = load_text(file_path)
    byte_chunks, masks = chunk_data(text_array, seq_len, window_size)
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

