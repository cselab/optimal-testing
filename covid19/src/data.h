#pragma once

#include <string>
#include <vector>

namespace epidemics {

/// A data value for the given region and day.
struct DataPoint {
    int day;
    int region;
    double value;
};

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
    std::vector<double> extComIu;  // Row-major undocumented infected foreign commuters [day][canton].
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
              std::vector<double> extComIu,
              std::vector<double> Ui);

    double getExternalCommutersIu(int day, int canton) const noexcept {
        int idx = day * (int)numRegions + canton;
        return idx < 0 || idx >= (int)extComIu.size() ? 0 : extComIu[idx];
    }

    void init();
};

/// UQ-specific data
struct ReferenceData {
    // Note: py/model.py:ReferenceData.to_cpp depends on this structure.

    /// List of known number of cases.
    std::vector<DataPoint> cases;

    /// Represent all known data point values into a single vector.
    /// Used for setting up Korali.
    std::vector<double> getReferenceData() const;

    /*
    /// Extract number of infected people for days and regions for which we
    /// have measured data, in the same order as the values returned by
    /// `getReferenceData()`.
    std::vector<double> getReferenceEvaluations(
            const std::vector<State> &states) const;
    */
};

ModelData readModelData(const char *filename = "data/cpp_model_data.dat");
ReferenceData readReferenceData(const char *filename = "data/cpp_reference_data.dat");

}  // namespace epidemics
