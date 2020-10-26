#include "bindings.h"
#include "cantons.h"
#include <base.h>
#include <seiin_interventions.h>

using namespace py::literals;

namespace epidemics {
namespace cantons {
namespace seiin_interventions {

template <typename T>
static auto exportParameters(py::module &m, const char *name) {
    auto pyParams = py::class_<Parameters<T>>(m, name)
        .def(py::init<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>(),
             "beta"_a,
             "mu"_a,
             "alpha"_a,
             "Z"_a,
             "D"_a,
             "theta"_a,
             "b1"_a,
             "b2"_a,
             "b3"_a,
             "d1"_a,
             "d2"_a,
             "d3"_a,
             "theta1"_a,
             "theta2"_a,
             "theta3"_a)
        .def_readwrite("beta", &Parameters<T>::beta)
        .def_readwrite("mu", &Parameters<T>::mu)
        .def_readwrite("alpha", &Parameters<T>::alpha)
        .def_readwrite("Z", &Parameters<T>::Z)
        .def_readwrite("D", &Parameters<T>::D)
        .def_readwrite("theta", &Parameters<T>::theta)
        .def_readwrite("b1", &Parameters<T>::b1)
        .def_readwrite("b2", &Parameters<T>::b2)
        .def_readwrite("b3", &Parameters<T>::b3)
        .def_readwrite("d1", &Parameters<T>::d1)
        .def_readwrite("d2", &Parameters<T>::d2)
        .def_readwrite("d3", &Parameters<T>::d3)
        .def_readwrite("theta1", &Parameters<T>::theta1)
        .def_readwrite("theta2", &Parameters<T>::theta2)
        .def_readwrite("theta3", &Parameters<T>::theta3)
        .def("__getitem__",
            [](const Parameters<T> &params, size_t index) {
                switch (index) {
                    case 0: return params.beta;
                    case 1: return params.mu;
                    case 2: return params.alpha;
                    case 3: return params.Z;
                    case 4: return params.D;
                    case 5: return params.theta;
                    case 6: return params.b1;
                    case 7: return params.b2;
                    case 8: return params.b3;
                    case 9: return params.d1;
                    case 10: return params.d2;
                    case 11: return params.d3;
                    case 12: return params.theta1;
                    case 13: return params.theta2;
                    case 14: return params.theta3;
                    default: throw py::index_error(std::to_string(index));
                }
            })
        .def("__len__", [](const Parameters<T> &) { return 15; });
    pyParams.attr("NUM_PARAMETERS") = 15;
    return pyParams;
}

template <typename T>
static auto exportState(py::module &m, const char *name) {
    return exportGenericState<State<T>>(m, name)
        .def("S", makeValuesGetter<State, T>(0), "Get a list of S for each canton.")
        .def("E", makeValuesGetter<State, T>(1), "Get a list of E for each canton.")
        .def("Ir", makeValuesGetter<State, T>(2), "Get a list of Ir for each canton.")
        .def("Iu", makeValuesGetter<State, T>(3), "Get a list of Iu for each canton.")
        .def("N", makeValuesGetter<State, T>(4), "Get a list of N for each canton.")
        .def("S", py::overload_cast<size_t>(&State<T>::S, py::const_), "Get S_i.")
        .def("E", py::overload_cast<size_t>(&State<T>::E, py::const_), "Get E_i.")
        .def("Ir", py::overload_cast<size_t>(&State<T>::Ir, py::const_), "Get Ir_i.")
        .def("Iu", py::overload_cast<size_t>(&State<T>::Iu, py::const_), "Get Iu_i.")
        .def("N", py::overload_cast<size_t>(&State<T>::N, py::const_), "Get N_i.");
}

void exportAll(py::module &/* top */, py::module &m) {
    exportParameters<double>(m, "Parameters");

    exportState<double>(m, "State");

    exportSolver<Solver, ModelData, State, Parameters>(m)
        .def("state_size", &Solver::stateSize, "Return the number of state variables.");
}

}  // namespace seiin_interventions
}  // namespace cantons
}  // namespace epidemics
