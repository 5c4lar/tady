from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tady.model.wcc_jax import wcc_label_propagation


@jax.jit
def neighbors_to_adj_matrix(neighbors):
    """
    Converts a neighbors tensor to an adjacency matrix, indicating edges
    between nodes.

    Args:
        neighbors: jnp.ndarray of shape (seq_len, neighbors). Contains the indices
                   of neighbors for each node, where -1 indicates no connection.

    Returns:
        adj_matrix: jnp.ndarray of shape (seq_len, seq_len). A matrix where a value of 1
                    indicates the presence of an edge between nodes, and 0 indicates no edge.
    """
    seq_len, max_neighbors = neighbors.shape

    # Create row indices corresponding to each node
    row_idx = jnp.arange(seq_len, dtype=np.int32)
    row_idx = jnp.broadcast_to(row_idx[:, None], (seq_len, max_neighbors))

    # Mask to identify valid connections (where neighbor index != -1)
    valid_mask = neighbors != -1

    # Initialize adjacency matrix as a matrix of zeros
    adj_matrix = jnp.zeros((seq_len, seq_len), dtype=bool)

    # Use scatter_add to set values in adj_matrix for valid connections
    adj_matrix = adj_matrix.at[row_idx, neighbors].set(valid_mask)

    return adj_matrix


@jax.jit
def bfs_reachability(adjacency_matrix):
    """
    Computes the reachability matrix using breadth-first search (BFS) instead of matrix powers.
    This is more efficient for sparse graphs.
    
    Args:
        adjacency_matrix: jnp.ndarray of shape (seq_len, seq_len), boolean adjacency matrix
                          
    Returns:
        reachability_matrix: jnp.ndarray of shape (seq_len, seq_len), boolean reachability matrix
    """
    n = adjacency_matrix.shape[0]
    
    # Initialize reachability matrix with the adjacency matrix
    # Also mark self-loops as reachable
    reachability = adjacency_matrix | jnp.eye(n, dtype=bool)
    
    # BFS-like propagation
    # At each step, if i can reach j and j can reach k, then i can reach k
    def body_fn(i, reach):
        return reach | (reach @ reach)
    
    # Log2(n) iterations are sufficient to cover all possible paths in a graph of size n
    num_iterations = max(1, int(np.ceil(np.log2(n))))
    reachability = jax.lax.fori_loop(0, num_iterations, body_fn, reachability)
    
    return reachability

@partial(jax.jit, static_argnames=("sliding_window"))
def get_attention(edges, sliding_window):
    '''
    Efficiently calculates attention mask based on graph reachability within sliding windows
    
    Args:
        edges: Tensor of shape (batch_size, sequence_length, neighbors) representing edge connections
        sliding_window: Tuple of (left_window, right_window) sizes
        
    Returns:
        attention_mask: Boolean tensor indicating which nodes can attend to which other nodes
    '''
    batch_size, sequence_length = edges.shape[:2]
    
    # Adjust edge indices to account for padding
    edges = jnp.where(edges < 0, -1, edges + sliding_window[0])
    edges = jnp.pad(edges, ((0, 0), sliding_window, (0, 0)),
                    mode="constant", constant_values=-1)
    
    # Process left side of the sliding window
    left_indices = jnp.arange(sliding_window[0] + 1, dtype=jnp.int32)[
        None, :] + jnp.arange(sequence_length, dtype=jnp.int32)[:, None]
    left_edges = edges[:, left_indices, :]
    
    # Adjust indices for the left side
    left_edges = left_edges - \
        jnp.arange(left_edges.shape[1], dtype=jnp.int32)[None, :, None, None]
    left_edges = jnp.where((left_edges >= (sliding_window[0] + 1)) | (
        left_edges < 0), -1, left_edges)
    
    # Convert edges to adjacency matrices
    left_adj_matrix = jax.vmap(
        jax.vmap(neighbors_to_adj_matrix))(left_edges)
    
    # Compute reachability using optimized BFS approach
    left_reachable = jax.vmap(
        jax.vmap(bfs_reachability))(left_adj_matrix)
    
    # Process right side of the sliding window
    right_indices = jnp.arange(sliding_window[1] + 1, dtype=jnp.int32)[
        None, :] + jnp.arange(sequence_length, dtype=jnp.int32)[:, None] + sliding_window[0]
    right_edges = edges[:, right_indices, :]
    
    # Adjust indices for the right side
    right_edges = right_edges - \
        jnp.arange(right_edges.shape[1], dtype=jnp.int32)[None, :, None, None]
    right_edges = jnp.where((right_edges >= (sliding_window[1] + 1)) | (
        right_edges < 0), -1, right_edges)
    
    # Convert edges to adjacency matrices and compute reachability
    right_adj_matrix = jax.vmap(
        jax.vmap(neighbors_to_adj_matrix))(right_edges)
    right_reachable = jax.vmap(
        jax.vmap(bfs_reachability))(right_adj_matrix)
    
    # Combine left and right reachability results to form the final attention mask
    attn_mask = jnp.concatenate(
        [
            (left_reachable[:, :, -1:, :-1] +
             left_reachable[:, :, :-1, -1:].transpose(0, 1, 3, 2)),
            jnp.ones((batch_size, sequence_length, 1, 1), dtype=bool),
            (right_reachable[:, :, :1, 1:] +
             right_reachable[:, :, 1:, :1].transpose(0, 1, 3, 2))
        ],
        axis=-1,
    )
    return attn_mask # Shape (batch_size, sequence_length, 1, 2 * window_size + 1)

@partial(jax.jit, static_argnames=("sliding_window"))
def get_attention_wccs(edges, sliding_window):
    """
    Computes the attention mask based on the edges and sliding window size.
    For each node, the mask indicates reachability to its k-th success neighbor (right part of the window)
    and k-th prev neighbors (left part of the window).
    This is to indicate in a superset disassemble sequence, which instructions are on the same non overlapping
    trace within the sliding window.
    For this variant, we first calculate the weakly connected components of the graph, the attention mask for each
    node is the ones belong to the same weakly connected component within the sliding window.
    return a boolean tensor of shape (batch_size, seq_len, 1, 2 * window_size + 1)
    """
    wcc_ids = wcc_label_propagation(edges, max_iterations=sliding_window[0]) # Shape (seq_len,)
    # Turn wcc_ids into slices of the sliding window, using dynamic slices
    # Get the start and end indices of each slice
    padded_wcc_ids = jnp.pad(wcc_ids, ((sliding_window[0], sliding_window[1])), mode="constant", constant_values=-1)
    slices = jax.vmap(lambda i: jax.lax.dynamic_slice_in_dim(padded_wcc_ids, i, sum(sliding_window) + 1, 0))(jnp.arange(edges.shape[0])) # Shape (seq_len, sum(sliding_window) + 1)
    
    attnention_mask = (slices == slices[:, sliding_window[0]:sliding_window[0]+1])
    return jnp.expand_dims(attnention_mask, axis=1)
    

@partial(jax.jit, static_argnames=("sliding_window"))
def get_attention_lite(edges, sliding_window):
    """
    Computes the attention mask based on the edges and sliding window size.
    For each node, the mask indicates reachability to its k-th success neighbor (right part of the window)
    and k-th prev neighbors (left part of the window).
    This is to indicate in a superset disassemble sequence, which instructions are on the same non overlapping
    trace within the sliding window.
    The attention mask is a boolean tensor of shape (seq_len, sum(sliding_window) + 1).
    Each node at position i can attend to reachable nodes in the range [i - left_window, i + right_window].
    
    Args:
        edges: jnp.ndarray of shape (seq_len, neighbors) representing the next positions of each node
                in the graph. The values are indices of the neighbors, where -1 indicates no connection.
                There are at most neighbors outgoing edges per node.
        sliding_window: Tuple of (left_window, right_window) sizes, where left_window == right_window.
        
    Returns:
        attention_mask: jnp.ndarray of shape (seq_len, 1, sum(sliding_window) + 1), boolean tensor indicating
                        which nodes can attend to which other nodes within the window.
    """
    # edges = edges[:, 3]
    window_size = sliding_window[0]  # Assumes left_window == right_window
    seq_len = edges.shape[0]

    # Initialize attention mask with False
    # Size of window: left_window + self + right_window = window_size + 1 + window_size
    attention_mask = jnp.zeros((seq_len, 2 * window_size + 1), dtype=bool)
    
    # Self-attention (center of the window)
    # The index for self-attention is window_size
    attention_mask = attention_mask.at[:, window_size].set(True)
    
    fwd_mask = jnp.zeros((seq_len, window_size), dtype=bool)
    
    # Helper function to compute forward reachability within window
    def compute_forward_reachability(idx, mask):
        # Initial state
        pos = idx
        curr_mask = mask
        # Track if we've found any new reachable nodes
        found_new = True
        step_count = 0
        
        # Forward traversal (right side of window)
        def forward_cond(state):
            _, _, found_new, step_count = state
            # Continue if we found new nodes in last iteration and haven't hit max steps
            return found_new & (step_count < window_size)
        
        def forward_body(state):
            pos, curr_mask, _, step_count = state
            next_pos = edges[pos]
            valid_next = (next_pos != -1) & (next_pos < seq_len) & (next_pos > idx) & (next_pos <= idx + window_size)
            # Only update position if valid
            new_pos = jnp.where(valid_next, next_pos, pos)
            # Check if this position was already marked
            already_marked = jnp.where(valid_next, curr_mask[next_pos - idx - 1], True)
            # Update mask if valid
            new_mask = jnp.where(valid_next, 
                                curr_mask.at[next_pos - idx - 1].set(True),
                                curr_mask)
            # We found new nodes if valid_next is True and position wasn't already marked
            found_new = valid_next & (~already_marked)
            return new_pos, new_mask, found_new, step_count + 1
        
        # Traverse forward until no new nodes or max steps reached
        _, fwd_mask, _, _ = jax.lax.while_loop(
            forward_cond,
            forward_body,
            (pos, curr_mask, found_new, step_count)
        )
        
        return fwd_mask
    
    # Apply the forward reachability computation for each node
    fwd_mask = jax.vmap(compute_forward_reachability)(jnp.arange(seq_len), fwd_mask)
    
    # Compute backward mask based on forward mask
    # If j can reach i in the forward pass, then i can be reached by j in the backward pass
    bwd_mask = jnp.zeros((seq_len, window_size), dtype=bool)
    
    # For each node i and each potential predecessor j within the window
    def compute_backward_mask(idx, mask):
        # For each position within the window before idx
        window_start = jnp.maximum(0, idx - window_size)
        
        # Calculate all positions j that are within the window before idx
        j_indices = jnp.arange(window_size)
        j_positions = window_start + j_indices
        
        # Filter to only include valid positions (j < idx)
        valid_j = j_positions < idx
        
        # For valid j positions, check if j can reach idx in the forward mask
        valid_j_positions = jnp.where(valid_j, j_positions, 0)
        
        # Where j can reach idx (using forward mask data)
        positions_in_fwd_mask = idx - valid_j_positions - 1
        reachable = jnp.where(valid_j, fwd_mask[valid_j_positions, positions_in_fwd_mask], False) # Shape (window_size,)
        
        # Set in backward mask
        positions_in_bwd_mask = window_size - (idx - valid_j_positions)
        mask = jnp.where(valid_j, reachable, mask)
        
        return mask
    
    # Apply the backward reachability computation for each node
    bwd_mask = jax.vmap(compute_backward_mask)(jnp.arange(seq_len), bwd_mask)
    
    # Combine forward and backward masks with self-attention
    attention_mask = attention_mask.at[:, :window_size].set(bwd_mask)
    attention_mask = attention_mask.at[:, window_size+1:].set(fwd_mask)
    
    return attention_mask.reshape((seq_len, 1, 2 * window_size + 1))


if __name__ == "__main__":
    test_edges = jnp.array([
        [1, -1, -1],  # Node 0: -> 1
        [2, -1, -1],  # Node 1: -> 2
        [0, -1, -1],  # Node 2: -> 0
        [4, -1, -1],  # Node 3: -> 4
        [-1, -1, -1], # Node 4: (no outgoing, connected from 3)
        [-1, -1, -1], # Node 5: (isolated)
        [-1, -1, -1]  # Node 6: (isolated)
    ], dtype=jnp.int32)
    print(get_attention_wccs(test_edges, (1, 1)))