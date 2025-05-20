import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=("max_iterations",))
def wcc_label_propagation(edges: jnp.ndarray, max_iterations: int = 100) -> jnp.ndarray:
    """
    Calculates Weakly Connected Components (WCC) for a graph using a label propagation algorithm.
    This version is structured to be more JIT-friendly by removing data-dependent early exits.

    The graph is defined by the `edges` array. Nodes are implicitly indexed from 0 to n-1.
    For WCC, an edge from node `i` to node `j` (i.e., `edges[i, k] == j`) means
    `i` and `j` are connected. The algorithm iteratively propagates the minimum
    label (initially the node's own index) within a component to all its members.

    Args:
        edges: A JAX array of shape `[n, m]`, where `n` is the number of nodes and `m`
               is the maximum number of outgoing edges for any node.
               `edges[i, k]` represents the k-th candidate outgoing edge of node `i`.
               A value of -1 indicates that this is not a valid edge.

    Returns:
        A JAX array of shape `[n,]`, where the i-th element is the WCC ID (label)
        for node `i`. Nodes within the same WCC will share the same integer label.
        These labels are derived from the node indices (typically the smallest
        node index in a component becomes its WCC ID).
    """
    n, m = edges.shape
    num_nodes = n

    # Initialize labels: each node starts in its own component, identified by its own index.
    # This is returned directly if num_nodes is 0.
    initial_labels = jnp.arange(num_nodes, dtype=jnp.int32)

    # Efficiently extract all valid directed edges (source, destination).
    # For WCC, these directed edges define undirected connections.
    row_indices = jnp.arange(num_nodes, dtype=jnp.int32)
    all_potential_sources_flat = jnp.repeat(row_indices, m)
    all_potential_dests_flat = edges.flatten()
    valid_edge_mask = (all_potential_dests_flat != -1)
    
    # --- Define functions for the lax.while_loop ---

    def cond_fun(state):
        _, changed, iteration = state
        return changed & (iteration < max_iterations)

    def body_fun(state):
        current_labels, changed, iteration = state
        iteration = iteration + 1

        labels_at_sources = current_labels[all_potential_sources_flat]
        labels_at_dests = current_labels[all_potential_dests_flat]
        
        min_candidate_labels_for_edge_pairs = jnp.where(valid_edge_mask, jnp.minimum(labels_at_sources, labels_at_dests), jnp.inf)

        next_labels = current_labels 
        next_labels = next_labels.at[all_potential_sources_flat].min(min_candidate_labels_for_edge_pairs)
        next_labels = next_labels.at[all_potential_dests_flat].min(min_candidate_labels_for_edge_pairs)
        
        changed = jnp.any(next_labels != current_labels)
        
        return next_labels, changed, iteration

    # --- End of lax.while_loop function definitions ---

    # Initial state for the loop. `changed` is True to ensure at least one iteration
    # if the loop is entered. If there are no edges, body_fun will set changed to False.
    loop_initial_state = (initial_labels, jnp.array(True, dtype=jnp.bool_), jnp.array(0, dtype=jnp.int32))

    # Execute the label propagation loop.
    # If source_nodes_actual is empty (no valid edges), the body_fun will execute once,
    # `next_labels` will equal `current_labels` (so `initial_labels`), `changed` will be False,
    # and the loop will terminate, returning `initial_labels`.
    final_labels, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_initial_state)
    
    return final_labels

if __name__ == "__main__":
    # Test Case Definition
    # Number of nodes (n) = 7
    # Max outgoing edges (m) = 3
    test_edges = jnp.array([
        [1, -1, -1],  # Node 0: -> 1
        [2, -1, -1],  # Node 1: -> 2
        [0, -1, -1],  # Node 2: -> 0
        [4, -1, -1],  # Node 3: -> 4
        [-1, -1, -1], # Node 4: (no outgoing, connected from 3)
        [-1, -1, -1], # Node 5: (isolated)
        [-1, -1, -1]  # Node 6: (isolated)
    ], dtype=jnp.int32)

    expected_wcc_ids = jnp.array([0, 0, 0, 3, 3, 5, 6], dtype=jnp.int32)

    # Running the function
    calculated_wcc_ids = wcc_label_propagation(test_edges) # Your function call

    print(f"Input edges:\n{test_edges}")
    print(f"Calculated WCC IDs: {calculated_wcc_ids}")
    print(f"Expected WCC IDs:   {expected_wcc_ids}")
    print(f"Test Passed: {jnp.array_equal(calculated_wcc_ids, expected_wcc_ids)}")