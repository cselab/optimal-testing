/// Code common to country and cantons models.

#pragma once

#include "bindings.h"
#include <integrator.h>

namespace epidemics {

IntegratorSettings integratorSettingsFromKwargs(py::kwargs kwargs);

template <typename Solver, typename State, typename Parameters, typename PySolver>
void exportSolverCommon(
        py::module &m,
        PySolver &pySolver,
        const char *name)
{
    using namespace py::literals;
    pySolver.def(
            name,
            [](const Solver &solver,
               const Parameters &params,
               State state,
               const std::vector<double> &tEval,
               py::kwargs kwargs)
            {
                SignalRAII breakRAII;
                return solver.solve(params, std::move(state), tEval,
                                    integratorSettingsFromKwargs(kwargs));
            }, "params"_a, "y0"_a, "t_eval"_a);
    pySolver.attr("model") = m;
}

/// Export a model Solver. Returns the Solver class handler.
template <typename Solver,
          typename ModelData,
          typename State,
          typename Parameters>
auto exportSolver(py::module &m) {
    using namespace py::literals;

    auto solver = py::class_<Solver>(m, "Solver")
        .def(py::init<ModelData>(), "model_data"_a)
        .def_property_readonly("model_data", &Solver::md)
        .def("solve_ad", [](const Solver &, py::args, py::kwargs) {
            throw std::runtime_error("'solve_ad' has been renamed to 'solve_params_ad'");
        });
    exportSolverCommon<Solver, State, Parameters>(m, solver, "solve");
    return solver;
}

/// Export a model State. Returns the State class handler.
template <typename State>
static auto exportGenericState(py::module &m, const char *name) {
    return py::class_<State>(m, name)
        .def(py::init<typename State::RawState>())
        .def("tolist", [](const State &state) {
            return state.raw();
        }, "Convert to a Python list of elements.")
        .def("__call__", [](const State &state, size_t index) {
            if (index < state.raw().size())
                return state.raw()[index];
            throw std::out_of_range(std::to_string(index));
        }, "Get state variables by index.")
        .def("__len__", &State::size, "Get the total number of state variables.");
}

}  // namespace epidemics
