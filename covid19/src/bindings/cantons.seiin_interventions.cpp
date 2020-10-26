#include "bindings.h"
#include "cantons.h"
#include <base.h>
#include <seiin_interventions.h>

using namespace py::literals;

namespace epidemics {
namespace cantons {
namespace seiin_interventions {

static auto exportParameters(py::module &m, const char *name) {
    auto pyParams = py::class_<Parameters>(m, name)
        .def(py::init<double, double, double, double, double,
                      double, double, double, double, double,
                      double, double, double, double, double>(),
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
        .def_readwrite("beta", &Parameters::beta)
        .def_readwrite("mu", &Parameters::mu)
        .def_readwrite("alpha", &Parameters::alpha)
        .def_readwrite("Z", &Parameters::Z)
        .def_readwrite("D", &Parameters::D)
        .def_readwrite("theta", &Parameters::theta)
        .def_readwrite("b1", &Parameters::b1)
        .def_readwrite("b2", &Parameters::b2)
        .def_readwrite("b3", &Parameters::b3)
        .def_readwrite("d1", &Parameters::d1)
        .def_readwrite("d2", &Parameters::d2)
        .def_readwrite("d3", &Parameters::d3)
        .def_readwrite("theta1", &Parameters::theta1)
        .def_readwrite("theta2", &Parameters::theta2)
        .def_readwrite("theta3", &Parameters::theta3)
        .def("__getitem__",
            [](const Parameters &params, size_t index) {
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
        .def("__len__", [](const Parameters &) { return 15; });
    pyParams.attr("NUM_PARAMETERS") = 15;
    return pyParams;
}

static auto exportState(py::module &m, const char *name) {
    return exportGenericState<State>(m, name)
        .def("S", makeValuesGetter<State>(0), "Get a list of S for each canton.")
        .def("E", makeValuesGetter<State>(1), "Get a list of E for each canton.")
        .def("Ir", makeValuesGetter<State>(2), "Get a list of Ir for each canton.")
        .def("Iu", makeValuesGetter<State>(3), "Get a list of Iu for each canton.")
        .def("N", makeValuesGetter<State>(4), "Get a list of N for each canton.")
        .def("S", py::overload_cast<size_t>(&State::S, py::const_), "Get S_i.")
        .def("E", py::overload_cast<size_t>(&State::E, py::const_), "Get E_i.")
        .def("Ir", py::overload_cast<size_t>(&State::Ir, py::const_), "Get Ir_i.")
        .def("Iu", py::overload_cast<size_t>(&State::Iu, py::const_), "Get Iu_i.")
        .def("N", py::overload_cast<size_t>(&State::N, py::const_), "Get N_i.");
}

void exportAll(py::module &/* top */, py::module &m) {
    exportParameters(m, "Parameters");

    exportState(m, "State");

    exportSolver<Solver, ModelData, State, Parameters>(m)
        .def("state_size", &Solver::stateSize, "Return the number of state variables.");
}

}  // namespace seiin_interventions
}  // namespace cantons
}  // namespace epidemics
