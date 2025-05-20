#include "graph_boost.hpp"
#include "llvm_disasm.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

py::array_t<uint32_t> find_wccs(py::array_t<int32_t> edges) {
  uint32_t num_nodes = edges.shape(0);
  auto edges_ptr = edges.data();
  auto node_wcc = PostDominatorTree::find_wccs(num_nodes, edges_ptr);
  py::array_t<uint32_t> wccs(num_nodes);
  auto wccs_ptr = wccs.mutable_data();
  std::memcpy(wccs_ptr, node_wcc.data(), num_nodes * sizeof(uint32_t));
  return wccs;
}

PYBIND11_MODULE(tady_cpp, m) {
  m.doc() = "Tady: A Neural Disassembler without Consistency Violations - "
            "Custom Graph Version";
  m.def("find_wccs", &find_wccs);
  py::class_<PostDominatorTree>(m, "PostDominatorTree")
      .def(py::init<py::array_t<int32_t>, py::array_t<bool>>(),
           py::arg("edges"), py::arg("cf_status"))
      .def(py::init<py::array_t<int32_t>, py::array_t<bool>,
                    py::array_t<uint32_t>, py::array_t<int32_t>>(),
           py::arg("edges"), py::arg("cf_status"), py::arg("wccs"),
           py::arg("dom_tree"))
      .def("prune", &PostDominatorTree::prune)
      .def("get_ipdom", &PostDominatorTree::get_ipdom)
      .def("get_errors", &PostDominatorTree::get_errors)
      .def("get_wccs", &PostDominatorTree::get_wccs)
      .def("get_components_size", &PostDominatorTree::get_components_size)
      .def("get_absolute_ipdom", &PostDominatorTree::get_absolute_ipdom);
  py::class_<Disassembler>(m, "Disassembler")
      .def(py::init<>())
      .def("superset_disasm", &Disassembler::superset_disasm)
      .def("flow_kind", &Disassembler::flow_kind)
      .def("disasm_to_str", &Disassembler::disasm_to_str)
      .def("linear_disasm", &Disassembler::linear_disasm)
      .def("inst_str", &Disassembler::inst_str);
}