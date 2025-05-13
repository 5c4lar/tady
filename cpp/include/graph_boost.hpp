#include <algorithm> // For std::vector construction from set
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dominator_tree.hpp>
#include <boost/graph/strong_components.hpp>
#include <map> // Keep for now, might be used elsewhere, though prune_tree won't use it.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <queue>
#include <set>
#include <vector>
using namespace boost;

using BiDiGraph = adjacency_list<vecS, vecS, bidirectionalS>;

namespace py = pybind11;

class PostDominatorTree {
  size_t num_nodes;
  const int32_t *edges;
  std::vector<float> weights;
  const bool *cf_status;
  BiDiGraph cfg;
  BiDiGraph dtree;
  uint32_t exit_node;
  std::vector<int32_t> ipdom;
  typedef boost::graph_traits<BiDiGraph>::vertex_descriptor Vertex;
  typedef boost::property_map<BiDiGraph, boost::vertex_index_t>::type IndexMap;
  typedef boost::iterator_property_map<std::vector<Vertex>::iterator, IndexMap>
      PredMap;

  uint32_t add_exit_node(BiDiGraph &cfg) {
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
        if (sccs[scc_idx].size() > 1 && !cf_status[v_node]) {
          continue;
        }
        boost::add_edge(v_node, exit_v, cfg);
      }
    }
    return exit_v;
  }

  std::tuple<BiDiGraph, uint32_t> build_reverse_dom_tree(BiDiGraph &cfg) {
    auto exit_v = add_exit_node(cfg); // Renamed 'exit' to 'exit_v'
    auto reverse_graph = boost::make_reverse_graph(cfg);
    std::vector<Vertex> domTreePredVector = std::vector<Vertex>(
        boost::num_vertices(reverse_graph), BiDiGraph::null_vertex());
    IndexMap indexMap(get(boost::vertex_index, reverse_graph));
    PredMap domTreePredMap =
        boost::make_iterator_property_map(domTreePredVector.begin(), indexMap);
    boost::lengauer_tarjan_dominator_tree(reverse_graph,
                                          boost::vertex(exit_v, reverse_graph),
                                          domTreePredMap); // Use exit_v
    BiDiGraph dtree_local =
        BiDiGraph(boost::num_vertices(reverse_graph)); // Renamed dtree
    for (auto v : boost::make_iterator_range(boost::vertices(reverse_graph))) {
      auto pred = domTreePredMap[v];
      if (pred != BiDiGraph::null_vertex()) {
        if (static_cast<size_t>(v) < ipdom.size()) {
          ipdom[v] = static_cast<int32_t>(pred);
        }
        boost::add_edge(pred, v, dtree_local);
      }
    }
    return std::make_tuple(dtree_local, exit_v); // Use exit_v and dtree_local
  }

  void propagate_weights(BiDiGraph &graph, uint32_t root) {
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

  std::vector<int32_t> prune_tree() {
    std::set<int32_t> remaining_nodes;
    if (boost::num_vertices(dtree) == 0) {
      return {};
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
        remaining_nodes.insert(current_node);

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
            if (child_node < this->num_nodes && cf_status[child_node]) {
              is_cf_child = true;
            }

            if (!is_cf_child) { // Fallthrough path
              // Original condition: current_node != root
              if (current_node != exit_node &&
                  weights[child_node] > max_weight_fallthrough.first) {
                max_weight_fallthrough = {weights[child_node], child_node};
              }
            } else { // CF path
              remaining_nodes.insert(child_node);
              q.push(child_node);
            }
          }
        }

        if (max_weight_fallthrough.second != static_cast<uint32_t>(-1)) {
          uint32_t fallthrough_child_node = max_weight_fallthrough.second;
          remaining_nodes.insert(fallthrough_child_node);
          q.push(fallthrough_child_node);
        }
      }
    }
    return std::vector<int32_t>(remaining_nodes.begin(), remaining_nodes.end());
  }


  std::map<std::string, std::set<uint32_t>> detect_errors() {
    // Create an empty pruned tree
    std::queue<int> q;
    q.push(exit_node);
    std::vector<bool> visited(num_vertices(dtree), false);
    std::map<std::string, std::set<uint32_t>> errors = {
        {"dangling", {}}, {"coexist", {}}, {"exclusive", {}}};
    std::vector<boost::default_color_type> colormap(num_vertices(dtree));
    struct dfs_visitor : public boost::default_dfs_visitor {
      std::map<std::string, std::set<uint32_t>> &errors;
      std::vector<bool> &visited;
      float *weights;
      std::vector<int32_t> &ipdom;
      dfs_visitor(std::map<std::string, std::set<uint32_t>> &e,
                  std::vector<bool> &vis, float *w, std::vector<int32_t> &ip)
          : errors(e), visited(vis), weights(w), ipdom(ip) {}
      void discover_vertex(const BiDiGraph::vertex_descriptor &v,
                           const BiDiGraph &g) {
        visited[v] = true;
        if (weights[v] > 0 && !errors["dangling"].contains(ipdom[v])) {
          errors["dangling"].emplace(v);
        }
      }
    } visitor(errors, visited, weights.data(), ipdom);
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
          if (!cf_status[child_node]) {
            if (node == exit_node) {
              boost::depth_first_visit(dtree, child_node, visitor,
                                       colormap.data());
            } else if (weights[child_node] > max_weight_fallthrough.first) {
              if (max_weight_fallthrough.first > 0.0) {
                errors["exclusive"].emplace(child_node);
                errors["exclusive"].emplace(max_weight_fallthrough.second);
              }
              max_weight_fallthrough = {weights[child_node], child_node};
            }
            continue;
          }
        }
      }
    }
    for (auto v : boost::make_iterator_range(boost::vertices(dtree))) {
      if (v != exit_node && weights[v] > 0 && !visited[v] && visited[ipdom[v]]) {
        errors["coexist"].emplace(v);
      }
    }
    return errors;
  }

public:
  PostDominatorTree(size_t n, const int32_t *e, const bool *cf = nullptr)
      : num_nodes(n), edges(e), weights(n + 1), cf_status(cf), ipdom(n),
        cfg(n) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        if (edges[i * 2 + j] != -1) {
          add_edge(i, edges[i * 2 + j], cfg);
        }
      }
    }
    std::tie(dtree, exit_node) = build_reverse_dom_tree(cfg);
  }
  PostDominatorTree(py::array_t<int32_t> edges, py::array_t<bool> cf_status)
      : PostDominatorTree(edges.shape(0), edges.data(), cf_status.data()) {
    // Constructor body is now empty as initialization is done via delegation
  }

  py::array_t<int32_t> get_ipdom() {
    return py::array_t<int32_t>(num_nodes, ipdom.data());
  }

  py::array_t<int32_t> prune(py::array_t<float> weights) {
    if (weights.size() != num_nodes) {
      throw std::invalid_argument("weights size must match num_nodes");
    }
    std::copy(weights.data(), weights.data() + num_nodes,
              this->weights.begin());
    propagate_weights(dtree, exit_node);
    auto pruned_nodes = prune_tree();
    return py::array_t<int32_t>(pruned_nodes.size(), pruned_nodes.data());
  }

  // Return Detect errors as dict of sets
  py::dict get_errors(py::array_t<float> weights) {
    if (weights.size() != num_nodes) {
      throw std::invalid_argument("weights size must match num_nodes");
    }
    std::copy(weights.data(), weights.data() + num_nodes,
              this->weights.begin());
    auto errors = detect_errors();
    py::dict result;
    for (const auto &[key, value] : errors) {
      auto array = py::array_t<int32_t>(value.size());
      std::copy(value.begin(), value.end(), array.mutable_data());
      result.attr("__setitem__")(key, array);
    }
    return result;
  }
};
// extern py::array_t<int32_t> process_graph_boost(py::array_t<int32_t> edges,
//                                                 py::array_t<float> weights,
//                                                 py::array_t<bool> cf_status);