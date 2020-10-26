#include "data.h"

namespace epidemics {

ModelData::ModelData(
        std::vector<std::string> regionKeys_,
        std::vector<double> Ni_,
        std::vector<double> Mij_):
    regionKeys(std::move(regionKeys_)),
    Ni(std::move(Ni_)),
    Mij(std::move(Mij_))
{
    init();
}

void ModelData::init() {
    size_t K = regionKeys.size();
    numRegions = K;
    nonzero_Mij.resize(numRegions);
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < K; ++j) {
            if (Mij[i * K + j] != 0 || Mij[i * K + j] != 0) {
                nonzero_Mij[i].push_back(j);
            }
        }
    }
}

}  // namespace epidemics
