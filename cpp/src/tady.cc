#include "disasm.h"
#include "llvm_disasm.hpp"
#include "graph.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(tady_cpp, m) {
  m.doc() = "Tady: A Neural Disassembler without Consistency Violations - "
            "Custom Graph Version";
  m.def("process_graph_pipeline", &process_graph_pipeline,
        "Process graph pipeline");
  m.def("ldasm", &disasm, "Disassemble x86/x64 code", py::arg("code"),
        py::arg("is64") = 1);
}