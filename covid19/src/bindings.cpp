#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/signal.h"

#include <boost/array.hpp>

#include "data.h"
#include "model.h"

namespace py = pybind11;
using namespace py::literals;

namespace epidemics {

namespace {

class SignalRAII {
public:
    SignalRAII() {
        check_signals_func = []() {
            // https://stackoverflow.com/questions/14707049/allowing-ctrl-c-to-interrupt-a-python-c-extension
            if (PyErr_CheckSignals() != 0)
                throw std::runtime_error("Signal received. Breaking.");
        };
    }
    ~SignalRAII() {
        check_signals_func = nullptr;
    }
};

}  // anonymous namespace

/// Helper function factory for State getters.
static auto makeValuesGetter(size_t valueIndex) {
    assert(0 <= valueIndex && valueIndex < State::kVarsPerRegion);
    /// Extract a subvector of the state corresponding to the given value.
    return [valueIndex](const State &state) {
        const double *p = state.raw().data();
        return std::vector<double>(p + valueIndex * state.numRegions(),
                                   p + (valueIndex + 1) * state.numRegions());
    };
}

static IntegratorSettings integratorSettingsFromKwargs(py::kwargs kwargs) {
    auto pop = kwargs.attr("pop");
    IntegratorSettings out;
    out.dt = pop("dt", out.dt).cast<double>();
    if (!kwargs.empty())
        throw py::key_error(kwargs.begin()->first.cast<std::string>());
    return out;
}

static void exportModelData(py::module &m) {
    py::class_<ModelData>(m, "ModelData")
        .def(py::init<std::vector<std::string>, std::vector<double>,
                      std::vector<double>, std::vector<double>,
                      std::vector<double>, std::vector<double>>(),
             "region_keys"_a, "Ni"_a, "Mij"_a, "Cij"_a,
             "ext_com_iu"_a, "Ui"_a)
        .def_readonly("Mij", &ModelData::Mij)
        .def_readonly("num_regions", &ModelData::numRegions);
}

static void exportParameters(py::module &m) {
    auto pyParams = py::class_<Parameters>(m, "Parameters")
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
}

static void exportState(py::module &m) {
    py::class_<State>(m, "State")
        .def(py::init<State::RawState>())
        .def("tolist", [](const State &state) {
            return state.raw();
        }, "Convert to a Python list of elements.")
        .def("__call__", [](const State &state, size_t index) {
            if (index < state.raw().size())
                return state.raw()[index];
            throw std::out_of_range(std::to_string(index));
        }, "Get state variables by index.")
        .def("__len__", &State::size, "Get the total number of state variables.")
        .def("S", makeValuesGetter(0), "Get a list of S for each canton.")
        .def("E", makeValuesGetter(1), "Get a list of E for each canton.")
        .def("Ir", makeValuesGetter(2), "Get a list of Ir for each canton.")
        .def("Iu", makeValuesGetter(3), "Get a list of Iu for each canton.")
        .def("N", makeValuesGetter(4), "Get a list of N for each canton.")
        .def("S", py::overload_cast<size_t>(&State::S, py::const_), "Get S_i.")
        .def("E", py::overload_cast<size_t>(&State::E, py::const_), "Get E_i.")
        .def("Ir", py::overload_cast<size_t>(&State::Ir, py::const_), "Get Ir_i.")
        .def("Iu", py::overload_cast<size_t>(&State::Iu, py::const_), "Get Iu_i.")
        .def("N", py::overload_cast<size_t>(&State::N, py::const_), "Get N_i.");
}

/// Export a model Solver. Returns the Solver class handler.
static void exportSolver(py::module &m) {
    using namespace py::literals;

    auto solve = [](
        const Solver &solver,
        const Parameters &params,
        State state,
        const std::vector<double> &tEval,
        py::kwargs kwargs)
    {
        SignalRAII breakRAII;
        return solver.solve(params, std::move(state), tEval,
                            integratorSettingsFromKwargs(kwargs));
    };

    py::class_<Solver>(m, "Solver")
        .def(py::init<ModelData>(), "model_data"_a)
        .def_property_readonly("model_data", &Solver::md)
        .def("solve", std::move(solve), "params"_a, "y0"_a, "t_eval"_a)
        .def("state_size", &Solver::stateSize, "Return the number of state variables.");
    // solver.attr("model") = m;
}

}  // namespace epidemics

PYBIND11_MODULE(libepidemics, m)
{
    using namespace epidemics;

    // Export the model.
    exportParameters(m);
    exportState(m);
    exportSolver(m);

    // Export model data (design parameters).
    exportModelData(m);
}
