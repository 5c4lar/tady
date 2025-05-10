from tady_cpp import process_graph_pipeline
   
if __name__ == "__main__":
    # Test the C++ extension
    import numpy as np
    import argparse
    import psutil
    import os
    import jax.numpy as jnp
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark graph memory consumption')
    parser.add_argument('--num_nodes', type=int, default=1000, help='Number of nodes in the graph')
    args = parser.parse_args()
    
    # Generate a random graph with the specified number of nodes
    num_nodes = args.num_nodes
    num_edges =  2  # Adjust as needed for desired density
    
    # Create random edges
    edges = np.random.randint(-1, num_nodes, (num_nodes, num_edges), dtype=np.int32)
    edges[:, 1] = -1
    
    # Create random weights and cf values
    weights = np.random.random(num_nodes).astype(np.float32)
    cf = np.random.choice([True, False], num_nodes)
    
    # Measure memory before graph creation
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # in MB
    print(f"Memory usage before graph creation: {mem_before:.2f} MB")
    graph = process_graph_pipeline(edges, weights, cf)
    memory_after = process.memory_info().rss / 1024 / 1024  # in MB
    print(f"Memory usage after graph creation: {memory_after:.2f} MB")
    print(f"Memory usage increase: {memory_after - mem_before:.2f} MB")
    print("Graph built successfully.")