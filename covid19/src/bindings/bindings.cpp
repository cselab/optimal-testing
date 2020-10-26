#include "bindings.h"

#include <data.h>
#include <integrator.h>

namespace py = pybind11;
using namespace py::literals;

namespace epidemics {

IntegratorSettings integratorSettingsFromKwargs(py::kwargs kwargs) {
    auto pop = kwargs.attr("pop");
    IntegratorSettings out;
    out.dt = pop("dt", out.dt).cast<double>();
    if (!kwargs.empty())
        throw py::key_error(kwargs.begin()->first.cast<std::string>());
    return out;
}

namespace cantons {

namespace seiin_interventions { void exportAll(py::module &top, py::module &m); }

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

}  // namespace cantons
}  // namespace epidemics

PYBIND11_MODULE(libepidemics, m)
{
    using namespace epidemics;
    py::class_<IntegratorSettings>(m, "IntegratorSettings")
        .def(py::init<double>(), "dt"_a)
        .def_readwrite("dt", &IntegratorSettings::dt);

    auto cantons = m.def_submodule("cantons");

    // Export the model.
    auto seiin_interventions = cantons.def_submodule("seiin_interventions");
    epidemics::cantons::seiin_interventions::exportAll(m, seiin_interventions);

    epidemics::cantons::exportModelData(cantons);
}
