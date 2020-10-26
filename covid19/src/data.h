#pragma once

#include <string>
#include <vector>

namespace epidemics {

/// Model bulk parameters (design parameters, not optimized for).
struct ModelData {
    // Note: The following files depend on the structure of this struct:
    //       epidemics/cantons/py/model.py:ModelData.to_cpp
    //       src/epidemics/bindings/cantons.template.cpp
    //       tests/py/common.py
    std::vector<std::string> regionKeys;
    std::vector<double> Ni;     // Region population.
    std::vector<double> Mij;    // Row-major migration matrix [to][from].
    std::vector<double> Cij;    // Row-major commute matrix [to][from].
    std::vector<double> Ui;     // User-defined

    // Computed.
    size_t numRegions;
    std::vector<double> invNi;  // 1 / region population.
    std::vector<double> C_plus_Ct;  // Cij + Cji.
    // Vector `nonzero_Mij[i]` contains indices `j`
    // for which `Mij[i][j] != 0` or  `Mij[j][i] != 0`.
    std::vector<std::vector<size_t>> nonzero_Mij;

    ModelData() = default;
    ModelData(std::vector<std::string> regionKeys,
              std::vector<double> Ni,
              std::vector<double> Mij,
              std::vector<double> Cij,
              std::vector<double> Ui);

    void init();
};

}  // namespace epidemics
