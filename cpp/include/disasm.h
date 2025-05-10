#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

extern py::array_t<uint8_t> disasm(const py::array_t<uint8_t> &code,
                                   bool is64 = 1);