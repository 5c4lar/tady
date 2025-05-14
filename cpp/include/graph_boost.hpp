#include <algorithm> // For std::vector construction from set
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/dominator_tree.hpp>
#include <boost/graph/strong_components.hpp>
#include <map> // Keep for now, might be used elsewhere, though prune_tree won't use it.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <queue>
#include <set>
#include <stdexcept>
#include <vector>
using namespace boost;

using BiDiGraph = adjacency_list<vecS, vecS, bidirectionalS>;

namespace py = pybind11;

class PostDominatorTree {
  size_t num_nodes;
  const int32_t *edges;
  const bool *cf_status;
  std::vector<int32_t> ipdom;
  std::vector<std::vector<uint32_t>> components;
  typedef boost::graph_traits<BiDiGraph>::vertex_descriptor Vertex;
  typedef boost::property_map<BiDiGraph, boost::vertex_index_t>::type IndexMap;
  typedef boost::iterator_property_map<std::vector<Vertex>::iterator, IndexMap>
      PredMap;

  uint32_t add_exit_node(BiDiGraph &cfg,
                         const std::vector<uint32_t> &component_nodes) {
    // Calculate SCCs and component mapping
    std::vector<int> component(boost::num_vertices(cfg));
    auto index_map = boost::get(boost::vertex_index, cfg);
    auto component_map_iter =
        boost::make_iterator_property_map(component.begin(), index_map);
    size_t num_sccs = boost::strong_components(cfg, component_map_iter);

    // Reconstruct SCCs (list of nodes per SCC)
    std::vector<std::vector<uint32_t>> sccs(num_sccs);
    for (auto v_desc : boost::make_iterator_range(boost::vertices(cfg))) {
      // Assuming vertex_descriptor (v_desc) is convertible to uint32_t
      sccs[component_map_iter[v_desc]].push_back(v_desc);
    }

    // Identify SCCs with no outgoing inter-SCC edges
    std::vector<bool> scc_has_inter_scc_outgoing_edge(num_sccs, false);
    if (num_sccs > 0) { // Avoid processing edges if no SCCs (e.g. empty graph)
      for (auto e : boost::make_iterator_range(boost::edges(cfg))) {
        auto u = boost::source(e, cfg);
        auto v_target = boost::target(e, cfg);
        if (component_map_iter[u] != component_map_iter[v_target]) {
          scc_has_inter_scc_outgoing_edge[component_map_iter[u]] = true;
        }
      }
    }

    std::vector<uint32_t>
        exit_scc_indices; // Stores indices of SCCs that are exits
    for (size_t i = 0; i < num_sccs; ++i) {
      if (!scc_has_inter_scc_outgoing_edge[i]) {
        exit_scc_indices.push_back(i);
      }
    }

    auto exit_v =
        boost::add_vertex(cfg); // Renamed 'exit' to 'exit_v' to avoid conflict
    for (auto scc_idx : exit_scc_indices) {
      for (auto v_node : sccs[scc_idx]) {
        // For the scc with multiple nodes (a loop structure),
        // only connect the control flow instruction to the exit node.
        // Assuming cf_status is indexed by original node id (v_node)
        if (sccs[scc_idx].size() > 1 && !cf_status[component_nodes[v_node]]) {
          continue;
        }
        boost::add_edge(v_node, exit_v, cfg);
      }
    }
    return exit_v;
  }

  // Helper function to find WCCs using Union-Find
  void find_wccs() {
    // Initialize Union-Find data structure
    std::vector<uint32_t> parent(num_nodes);
    std::vector<uint32_t> rank(num_nodes, 0);
    for (uint32_t i = 0; i < num_nodes; ++i) {
      parent[i] = i;
    }

    // Find function with path compression
    auto find = [&parent](uint32_t x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };

    // Union function with rank optimization
    auto unite = [&parent, &rank, &find](uint32_t x, uint32_t y) {
      x = find(x);
      y = find(y);
      if (x == y)
        return;
      if (rank[x] < rank[y]) {
        parent[x] = y;
      } else if (rank[x] > rank[y]) {
        parent[y] = x;
      } else {
        parent[y] = x;
        rank[x]++;
      }
    };

    // Process all edges to build WCCs
    for (size_t i = 0; i < num_nodes; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        int32_t target = edges[i * 2 + j];
        if (target != -1) {
          unite(i, static_cast<uint32_t>(target));
        }
      }
    }

    // Group nodes by their component
    std::map<uint32_t, std::vector<uint32_t>> component_map;
    for (uint32_t i = 0; i < num_nodes; ++i) {
      component_map[find(i)].push_back(i);
    }

    // Convert map to vector of components
    components.reserve(component_map.size());
    for (const auto &[_, nodes] : component_map) {
      components.push_back(nodes);
    }
  }

  // Helper function to build post dominator tree for a single component
  void build_component_dom_tree(const std::vector<uint32_t> &component_nodes) {
    // Create a subgraph for this component
    BiDiGraph component_graph(component_nodes.size());
    std::map<uint32_t, uint32_t>
        node_mapping; // Maps original node IDs to component-local IDs

    // Create node mapping
    for (size_t i = 0; i < component_nodes.size(); ++i) {
      node_mapping[component_nodes[i]] = i;
    }

    // Add edges for this component
    for (size_t i = 0; i < component_nodes.size(); ++i) {
      uint32_t orig_node = component_nodes[i];
      for (size_t j = 0; j < 2; ++j) {
        int32_t target = edges[orig_node * 2 + j];
        if (target != -1) {
          add_edge(i, node_mapping[target], component_graph);
        }
      }
    }

    // Add exit node for this component
    auto component_exit = add_exit_node(component_graph, component_nodes);
    auto reverse_graph = boost::make_reverse_graph(component_graph);

    // Calculate dominator tree for this component
    std::vector<Vertex> domTreePredVector(boost::num_vertices(reverse_graph),
                                          BiDiGraph::null_vertex());
    IndexMap indexMap(get(boost::vertex_index, reverse_graph));
    PredMap domTreePredMap =
        boost::make_iterator_property_map(domTreePredVector.begin(), indexMap);

    boost::lengauer_tarjan_dominator_tree(
        reverse_graph, boost::vertex(component_exit, reverse_graph),
        domTreePredMap);

    // Create dominator tree for this component
    for (auto v : boost::make_iterator_range(boost::vertices(reverse_graph))) {
      if (v == component_exit) {
        continue;
      }
      ipdom[component_nodes[v]] = domTreePredMap[v];
    }
  }

  std::tuple<BiDiGraph, uint32_t>
  recover_component_dom_tree(const std::vector<uint32_t> &component_nodes) {
    // Witht the calculated ipdom, reconstruct the dominator tree.
    BiDiGraph dtree(component_nodes.size());
    uint32_t exit_node = component_nodes.size();

    for (size_t i = 0; i < component_nodes.size(); ++i) {
      if (ipdom[component_nodes[i]] != -1) {
        add_edge(ipdom[component_nodes[i]], i, dtree);
      }
    }
    return std::make_tuple(dtree, exit_node);
  }

  void propagate_weights(BiDiGraph &graph, uint32_t root,
                         std::vector<float> &weights) {
    // Post-order DFS to calculate subtree weights
    std::vector<boost::default_color_type> colormap(num_vertices(graph));
    struct dfs_visitor : public boost::default_dfs_visitor {
      // Dictionary to store maximum weight of subtree rooted at each node
      float *weights;
      dfs_visitor(float *w) : weights(w) {}
      void finish_vertex(const BiDiGraph::vertex_descriptor &v,
                         const BiDiGraph &g) {
        double total_weight = weights[v];

        // Sum the weights of valid child subtrees
        double child_subtree_weight = 0;
        for (auto child : boost::make_iterator_range(boost::out_edges(v, g))) {
          int child_node = boost::target(child, g);
          if (weights[child_node] > 0) {
            child_subtree_weight += weights[child_node];
          }
        }
        // Calculate the weight of the subtree rooted at this node
        weights[v] = std::max(0.0, total_weight + child_subtree_weight);
      }
    } visitor(weights.data());
    boost::depth_first_search(graph, visitor, colormap.data(), root);
  }

  void prune_tree(const std::vector<uint32_t> &component_nodes,
                  std::set<uint32_t> &remaining_nodes,
                  std::vector<float> &weights) {
    auto [dtree, exit_node] = recover_component_dom_tree(component_nodes);
    propagate_weights(dtree, exit_node, weights);
    if (boost::num_vertices(dtree) == 0) {
      return;
    }

    std::queue<int32_t> q;
    q.push(exit_node);

    while (!q.empty()) {
      int32_t current_node = q.front();
      q.pop();

      bool current_node_has_positive_weight = false;
      if (current_node < weights.size() && weights[current_node] > 0) {
        current_node_has_positive_weight = true;
      }

      // Node is processed if it's the root OR it has positive weight.
      if (current_node_has_positive_weight || current_node == exit_node) {
        if (current_node != exit_node) {
          remaining_nodes.insert(component_nodes[current_node]);
        }

        std::pair<float, uint32_t> max_weight_fallthrough = {
            0.0, static_cast<uint32_t>(-1)}; // Use uint32_t for ID

        for (auto edge_desc : boost::make_iterator_range(
                 boost::out_edges(current_node, dtree))) {
          int32_t child_node = boost::target(edge_desc, dtree);

          bool child_has_positive_weight = false;
          if (child_node < weights.size() && weights[child_node] > 0) {
            child_has_positive_weight = true;
          }
          // If child_node >= weights.size(), its weight is effectively not
          // positive for pruning.

          if (child_has_positive_weight) {
            bool is_cf_child = false;
            // cf_status is for original nodes (0 to num_nodes-1)
            // exit_node (if child_node is it) is considered non-CF.
            if (child_node < this->num_nodes &&
                cf_status[component_nodes[child_node]]) {
              is_cf_child = true;
            }

            if (!is_cf_child) { // Fallthrough path
              // Original condition: current_node != root
              if (current_node != exit_node &&
                  weights[child_node] > max_weight_fallthrough.first) {
                max_weight_fallthrough = {weights[child_node], child_node};
              }
            } else { // CF path
              remaining_nodes.insert(component_nodes[child_node]);
              q.push(child_node);
            }
          }
        }

        if (max_weight_fallthrough.second != static_cast<uint32_t>(-1)) {
          uint32_t fallthrough_child_node = max_weight_fallthrough.second;
          remaining_nodes.insert(component_nodes[fallthrough_child_node]);
          q.push(fallthrough_child_node);
        }
      }
    }
  }

  void detect_errors(const std::vector<uint32_t> &component_nodes,
                     std::map<std::string, std::set<uint32_t>> &errors,
                     std::vector<float> &weights) {
    // Create an empty pruned tree
    auto [dtree, exit_node] = recover_component_dom_tree(component_nodes);
    std::queue<int> q;
    q.push(exit_node);
    std::vector<bool> visited(num_vertices(dtree), false);
    std::vector<boost::default_color_type> colormap(num_vertices(dtree));
    struct dfs_visitor : public boost::default_dfs_visitor {
      std::map<std::string, std::set<uint32_t>> &errors;
      std::vector<bool> &visited;
      float *weights;
      std::vector<int32_t> &ipdom;
      const std::vector<uint32_t> &component_nodes;
      dfs_visitor(std::map<std::string, std::set<uint32_t>> &e,
                  std::vector<bool> &vis, float *w, std::vector<int32_t> &ip,
                  const std::vector<uint32_t> &cn)
          : errors(e), visited(vis), weights(w), ipdom(ip),
            component_nodes(cn) {}
      void discover_vertex(const BiDiGraph::vertex_descriptor &v,
                           const BiDiGraph &g) {
        visited[v] = true;
        if (weights[v] > 0 && !visited[ipdom[component_nodes[v]]]) {
          errors["dangling"].emplace(component_nodes[v]);
        }
      }
    } visitor(errors, visited, weights.data(), ipdom, component_nodes);
    while (!q.empty()) {
      int node = q.front();
      q.pop();
      if ((weights[node] > 0) || (node == exit_node)) {
        std::pair<float, int> max_weight_fallthrough = {0.0, -1};
        for (auto child :
             boost::make_iterator_range(boost::out_edges(node, dtree))) {
          int child_node = boost::target(child, dtree);
          visited[child_node] = true;
          if (weights[child_node] > 0) {
            q.push(child_node);
          }
          if (!cf_status[component_nodes[child_node]]) {
            if (node == exit_node) {
              boost::depth_first_visit(dtree, child_node, visitor,
                                       colormap.data());
            } else if (weights[child_node] > max_weight_fallthrough.first) {
              if (max_weight_fallthrough.first > 0.0) {
                errors["exclusive"].emplace(component_nodes[child_node]);
                errors["exclusive"].emplace(
                    component_nodes[max_weight_fallthrough.second]);
              }
              max_weight_fallthrough = {weights[child_node], child_node};
            }
            continue;
          }
        }
      }
    }
    for (auto v : boost::make_iterator_range(boost::vertices(dtree))) {
      if (v != exit_node && weights[v] > 0 && !visited[v] &&
          visited[ipdom[component_nodes[v]]]) {
        errors["coexist"].emplace(component_nodes[v]);
      }
    }
  }

public:
  PostDominatorTree(size_t n, const int32_t *e, const bool *cf = nullptr)
      : num_nodes(n), edges(e), cf_status(cf), ipdom(n, -1) {
    // Find weakly connected components directly from edges
    find_wccs();

    // Process each component separately
    for (const auto &component_nodes : components) {
      if (!component_nodes.empty()) {
        build_component_dom_tree(component_nodes);
      }
    }
  }
  PostDominatorTree(py::array_t<int32_t> edges, py::array_t<bool> cf_status)
      : PostDominatorTree(edges.shape(0), edges.data(), cf_status.data()) {
    // Constructor body is now empty as initialization is done via delegation
  }

  py::array_t<int32_t> get_ipdom() {
    return py::array_t<int32_t>(num_nodes, ipdom.data());
  }

  py::array_t<int32_t> get_components_size() {
    py::array_t<int32_t> sizes(components.size());
    for (size_t i = 0; i < components.size(); ++i) {
      sizes.mutable_at(i) = components[i].size();
    }
    return sizes;
  }

  py::array_t<int32_t> prune(py::array_t<float> weights) {
    if (weights.size() != num_nodes) {
      throw std::invalid_argument("weights size must match num_nodes");
    }
    const float *weights_data = weights.data();
    std::set<uint32_t> pruned_nodes;
    for (const auto &component_nodes : components) {
      std::vector<float> component_weights(component_nodes.size() + 1);
      for (size_t i = 0; i < component_nodes.size(); ++i) {
        component_weights[i] = weights_data[component_nodes[i]];
      }
      component_weights[component_nodes.size()] = 0;
      prune_tree(component_nodes, pruned_nodes, component_weights);
    }
    auto array = py::array_t<int32_t>(pruned_nodes.size());
    std::copy(pruned_nodes.begin(), pruned_nodes.end(), array.mutable_data());
    return array;
  }

  // Return Detect errors as dict of sets
  py::dict get_errors(py::array_t<float> weights) {
    if (weights.size() != num_nodes) {
      throw std::invalid_argument("weights size must match num_nodes");
    }
    const float *weights_data = weights.data();
    std::map<std::string, std::set<uint32_t>> errors = {
        {"dangling", {}}, {"coexist", {}}, {"exclusive", {}}};
    for (const auto &component_nodes : components) {
      std::vector<float> component_weights(component_nodes.size() + 1);
      for (size_t i = 0; i < component_nodes.size(); ++i) {
        component_weights[i] = weights_data[component_nodes[i]];
      }
      component_weights[component_nodes.size()] = 0;
      detect_errors(component_nodes, errors, component_weights);
    }
    py::dict result;
    for (const auto &[key, value] : errors) {
      auto array = py::array_t<int32_t>(value.size());
      std::copy(value.begin(), value.end(), array.mutable_data());
      result.attr("__setitem__")(key, array);
    }
    return result;
  }
};
