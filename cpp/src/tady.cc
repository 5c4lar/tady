#include "graph_boost.hpp"
#include "llvm_disasm.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(tady_cpp, m) {
  m.doc() = "Tady: A Neural Disassembler without Consistency Violations - "
            "Custom Graph Version";
  py::class_<PostDominatorTree>(m, "PostDominatorTree")
      .def(py::init<py::array_t<int32_t>, py::array_t<bool>>(),
           py::arg("edges"), py::arg("cf_status"))
      .def("prune", &PostDominatorTree::prune)
      .def("get_ipdom", &PostDominatorTree::get_ipdom)
      .def("get_errors", &PostDominatorTree::get_errors)
      .def("get_components_size", &PostDominatorTree::get_components_size);
  py::class_<Disassembler>(m, "Disassembler")
      .def(py::init<>())
      .def("superset_disasm", &Disassembler::superset_disasm)
      .def("flow_kind", &Disassembler::flow_kind)
      .def("disasm_to_str", &Disassembler::disasm_to_str)
      .def("linear_disasm", &Disassembler::linear_disasm);
}