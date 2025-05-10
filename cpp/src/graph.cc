#include <algorithm>  // For std::fill, std::sort (optional in SCC/WCC output)
#include <cstdint>    // For uint32_t
#include <functional> // For std::function used in tree traversal
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stack>
#include <unordered_map> // For original_node_to_subgraph_node_map
#include <unordered_set>
#include <vector>
#include <thread>

namespace py = pybind11;


class Graph {

public:
  size_t num_nodes;
  const int32_t *edges;
  const float *weights;
  const bool *cf_status;
  using vertex_t = uint32_t; // Suitable for up to ~4 billion nodes
  using scc_id_t = int32_t;
  // using wcc_t = std::vector<vertex_t>;
  using scc_t = std::vector<vertex_t>;
  // std::vector<wcc_t> wccs;
  std::vector<scc_t> sccs;
  std::vector<scc_id_t> node_to_scc_id; // Map from node to SCC ID
  std::vector<scc_id_t> ipdom; // immediate post-dominator
  std::vector<std::vector<scc_id_t>> in_edges;
  std::vector<std::vector<scc_id_t>> out_edges; // Added for post-dominator calculation
  Graph(size_t n, const int32_t *e, const float *w, const bool *cf)
      : num_nodes(n), edges(e), weights(w), cf_status(cf), node_to_scc_id(n, -1) {
        // compute_wccs();
        compute_sccs();
        compute_condensation_edges();
        compute_ipdom();
      }

  // void compute_wccs() {
  //   std::vector<vertex_t> parent(num_nodes);
  //   std::iota(parent.begin(), parent.end(), 0); // init parent

  //   // Path compression find
  //   auto find = [&](vertex_t u) {
  //     while (u != parent[u]) {
  //       parent[u] = parent[parent[u]];
  //       u = parent[u];
  //     }
  //     return u;
  //   };

  //   // Union two components
  //   auto unite = [&](vertex_t u, vertex_t v) {
  //     vertex_t ru = find(u);
  //     vertex_t rv = find(v);
  //     if (ru != rv)
  //       parent[ru] = rv;
  //   };

  //   for (vertex_t u = 0; u < num_nodes; ++u) {
  //     int32_t v0 = edges[2 * u];
  //     int32_t v1 = edges[2 * u + 1];
  //     if (v0 > 0) {
  //       unite(u, v0);
  //     }
  //     if (v1 > 0) {
  //       unite(u, v1);
  //     }
  //   }

  //   // Final flatten
  //   for (vertex_t u = 0; u < num_nodes; ++u) {
  //     parent[u] = find(u);
  //   }
  //   // Create mapping from parent to component ID
  //   std::unordered_map<vertex_t, size_t> parent_to_id;

  //   for (vertex_t u = 0; u < num_nodes; ++u) {
  //     vertex_t p = parent[u];
  //     if (parent_to_id.find(p) == parent_to_id.end()) {
  //       parent_to_id[p] = wccs.size();
  //       wccs.push_back({});
  //     }
  //     wccs[parent_to_id[p]].push_back(u);
  //   }
  // }
  void compute_sccs() {
    std::vector<scc_id_t> index(num_nodes, -1);
    std::vector<int> lowlink(num_nodes, -1);
    std::vector<bool> on_stack(num_nodes, false);
    std::stack<vertex_t> S;
    int current_index = 0;

    std::function<void(vertex_t)> strongconnect =
        [&](vertex_t v) {
          index[v] = lowlink[v] = current_index++;
          S.push(v);
          on_stack[v] = true;

          for (int i = 0; i < 2; ++i) {
            int32_t w = edges[2 * v + i];
            // Skip invalid edges (-1) or out of bounds indices
            if (w >= 0 && w < num_nodes) {
              if (index[w] == -1) {
                strongconnect(w);
                lowlink[v] = std::min(lowlink[v], lowlink[w]);
              } else if (on_stack[w]) {
                lowlink[v] = std::min(lowlink[v], index[w]);
              }
            }
          }

          if (lowlink[v] == index[v]) {
            scc_t component;
            vertex_t w;
            scc_id_t scc_id = sccs.size(); // Current SCC ID
            do {
              w = S.top();
              S.pop();
              on_stack[w] = false;
              component.push_back(w);
              node_to_scc_id[w] = scc_id; // Assign SCC ID to the node
            } while (w != v);
            sccs.push_back(std::move(component));
          }
        };

    for (vertex_t v = 0; v < num_nodes; ++v) {
      if (index[v] == -1) {
        strongconnect(v);
      }
    }
  }

  void compute_condensation_edges() {
    in_edges.resize(sccs.size());
    out_edges.resize(sccs.size());
    
    for (vertex_t v = 0; v < num_nodes; ++v) {
      scc_id_t v_condensation_node_id = node_to_scc_id[v]; // Use node_to_scc_id instead of index
      
      // Skip nodes that weren't assigned to an SCC
      if (v_condensation_node_id == -1) {
        continue;
      }
      
      for (int i = 0; i < 2; ++i) {
        int32_t w = edges[2 * v + i];
        
        // Ensure w is a valid node index
        if (w >= 0 && w < num_nodes) {
          scc_id_t w_condensation_node_id = node_to_scc_id[w]; // Use node_to_scc_id instead of index
          
          // Skip nodes not in SCCs or in the same SCC as v
          if (w_condensation_node_id != -1 && 
              w_condensation_node_id != v_condensation_node_id) {
            in_edges[w_condensation_node_id].push_back(v_condensation_node_id);
            out_edges[v_condensation_node_id].push_back(w_condensation_node_id);
          }
        }
      }
    }
  }
  void compute_ipdom() {
    // Compute the immediate post-dominator of each condensation node
    // Using Lengauer–Tarjan algorithm
    // ipdom[i] is the immediate post-dominator of node i
    // ipdom[i] is the node j such that j is the immediate post-dominator of i
    // if there is no such node, ipdom[i] is -1

    // Initialize arrays for the condensation graph
    size_t num_condensation_nodes = sccs.size();
    
    // Initialize ipdom vector
    ipdom.resize(num_condensation_nodes, -1);
    
    // Early exit if no nodes
    if (num_condensation_nodes == 0) {
      return;
    }
    
    // Create a dummy exit node for post-dominators
    scc_id_t dummy_exit = num_condensation_nodes;
    
    // Find all nodes with zero out-degree in the condensation graph
    std::vector<scc_id_t> exit_nodes;
    for (scc_id_t i = 0; i < num_condensation_nodes; ++i) {
      if (out_edges[i].empty()) {
        exit_nodes.push_back(i);
      }
    }
    
    // Build parent tree with DFS traversal
    std::vector<scc_id_t> parent(num_condensation_nodes + 1, -1);
    std::vector<bool> visited(num_condensation_nodes + 1, false);
    std::vector<scc_id_t> dfs_order;
    std::vector<scc_id_t> dfs_num(num_condensation_nodes + 1, -1);
    
    std::function<void(scc_id_t)> dfs = [&](scc_id_t v) {
      visited[v] = true;
      dfs_num[v] = dfs_order.size();
      dfs_order.push_back(v);
      
      if (v == dummy_exit) {
        // For dummy exit node, its children are the exit nodes
        for (scc_id_t exit_node : exit_nodes) {
          if (!visited[exit_node]) {
            parent[exit_node] = v;
            dfs(exit_node);
          }
        }
      } else {
        // For regular nodes, use in_edges (which are out edges in the post-dominator tree)
        for (scc_id_t u : in_edges[v]) {
          if (!visited[u]) {
            parent[u] = v;
            dfs(u);
          }
        }
      }
    };
    
    // Start DFS from the dummy exit node
    dfs(dummy_exit);
    
    // Set parent of dummy_exit to itself
    parent[dummy_exit] = dummy_exit;
    
    // If not all nodes were visited in DFS, the graph might have multiple components
    // We can still proceed with partial results
    
    std::vector<scc_id_t> ancestor(num_condensation_nodes + 1, -1);
    std::vector<scc_id_t> label(num_condensation_nodes + 1);
    std::vector<scc_id_t> semi(num_condensation_nodes + 1);
    std::vector<scc_id_t> size(num_condensation_nodes + 1, 1);
    std::vector<scc_id_t> child(num_condensation_nodes + 1, -1);
    std::vector<std::vector<scc_id_t>> bucket(num_condensation_nodes + 1);

    // Initialize label and semi arrays
    for (scc_id_t i = 0; i <= num_condensation_nodes; ++i) {
      label[i] = i;
      semi[i] = dfs_num[i] != -1 ? dfs_num[i] : (num_condensation_nodes + 1); // Large value for unvisited nodes
    }

    // Step 1: Compute semi-dominators
    std::function<void(scc_id_t)> compress = [&](scc_id_t v) {
      if (ancestor[v] != -1 && ancestor[ancestor[v]] != -1) {
        compress(ancestor[v]);
        if (semi[label[ancestor[v]]] < semi[label[v]]) {
          label[v] = label[ancestor[v]];
        }
        ancestor[v] = ancestor[ancestor[v]];
      }
    };

    std::function<scc_id_t(scc_id_t)> eval = [&](scc_id_t v) {
      if (ancestor[v] == -1) return v;
      compress(v);
      return label[v];
    };

    auto link = [&](scc_id_t v, scc_id_t w) {
      if (v == -1 || w == -1) return; // Skip invalid nodes
      
      scc_id_t s = w;
      
      // Make sure child[s] is valid before checking semi values
      if (child[s] == -1) {
        // No action needed
      } else {
        while (child[s] != -1 && semi[label[w]] < semi[label[child[s]]]) {
          if (child[child[s]] != -1 && size[s] + size[child[child[s]]] >= 2 * size[child[s]]) {
            ancestor[child[s]] = s;
            child[s] = child[child[s]];
          } else {
            size[child[s]] = size[s];
            ancestor[s] = child[s];
            s = child[s];
          }
        }
      }
      
      label[s] = label[w];
      size[v] += size[w];
      if (size[v] < 2 * size[w]) {
        std::swap(s, child[v]);
      }
      while (s != -1) {
        ancestor[s] = v;
        s = child[s];
      }
    };

    // Process nodes in reverse DFS order
    for (int i = dfs_order.size() - 1; i > 0; --i) {
      scc_id_t w = dfs_order[i];
      
      // Process predecessors in the post-dominator tree
      if (w == dummy_exit) {
        // Dummy exit has exit nodes as predecessors
        for (scc_id_t u : exit_nodes) {
          if (dfs_num[u] == -1) continue; // Skip nodes not in the DFS tree
          
          scc_id_t u_semi = eval(u);
          if (semi[u_semi] < semi[w]) {
            semi[w] = semi[u_semi];
          }
        }
      } else {
        // Regular nodes have out_edges as predecessors
        for (scc_id_t u : out_edges[w]) {
          if (dfs_num[u] == -1) continue; // Skip nodes not in the DFS tree
          
          scc_id_t u_semi = eval(u);
          if (semi[u_semi] < semi[w]) {
            semi[w] = semi[u_semi];
          }
        }
      }
      
      bucket[dfs_order[semi[w]]].push_back(w);
      link(parent[w], w);

      // Process bucket[parent[w]]
      if (parent[w] != -1) {
        for (scc_id_t v : bucket[parent[w]]) {
          scc_id_t u = eval(v);
          ipdom[v] = (semi[u] < semi[v]) ? u : parent[w];
        }
        bucket[parent[w]].clear();
      }
    }

    // Step 2: Compute immediate dominators
    for (size_t i = 1; i < dfs_order.size(); ++i) {
      scc_id_t w = dfs_order[i];
      if (w == dummy_exit) continue; // Skip the dummy exit node
      if (w >= num_condensation_nodes) continue; // Skip if out of bounds
      
      if (ipdom[w] != dfs_order[semi[w]]) {
        if (ipdom[w] != -1 && ipdom[w] < num_condensation_nodes) {
          ipdom[w] = ipdom[ipdom[w]];
        }
      }
      
      // Special case: if ipdom is the dummy exit, set to -1 (no post-dominator)
      if (ipdom[w] == dummy_exit) {
        ipdom[w] = -1;
      }
    }
  }
};

int process_graph_pipeline(py::array_t<int32_t> edges,
                           py::array_t<float> weights,
                           py::array_t<bool> cf_status) {
  // Validate that edges has expected shape (num_nodes, 2)
  if (edges.ndim() != 2 || edges.shape(1) != 2) {
    throw std::runtime_error("Edges array must be a 2D array with shape (num_nodes, 2)");
  }
  
  // Edges is a 2D array of shape (num_edges, 2)
  size_t num_nodes = edges.shape(0);
  const int32_t *edges_ptr = edges.data();
  
  // Validate weights and cf_status sizes match the number of nodes
  if (weights.size() < num_nodes) {
    throw std::runtime_error("Weights array size must be at least equal to number of nodes");
  }
  
  if (cf_status.size() < num_nodes) {
    throw std::runtime_error("CF status array size must be at least equal to number of nodes");
  }
  
  const float *weights_ptr = weights.data();
  const bool *cf_status_ptr = cf_status.data();
  
  Graph graph(num_nodes, edges_ptr, weights_ptr, cf_status_ptr);
  // graph.process_wccs();
  return graph.sccs.size();
}
