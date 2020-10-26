#pragma once

#include <string>
#include <vector>

namespace epidemics {

struct ModelData {
    std::vector<std::string> regionKeys;
    std::vector<double> Ni;     // Region population.
    std::vector<double> Mij;    // Row-major connections' matrix [to][from].

    // Computed.
    size_t numRegions;
    std::vector<std::vector<size_t>> nonzero_Mij;

    ModelData() = default;
    ModelData(std::vector<std::string> regionKeys,
              std::vector<double> Ni,
              std::vector<double> Mij);

    void init();
};

}  // namespace epidemics
