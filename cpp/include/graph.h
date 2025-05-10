#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
extern int process_graph_pipeline(py::array_t<int32_t> edges,
                                  py::array_t<float> weights,
                                  py::array_t<bool> cf_status);