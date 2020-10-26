#include "data.h"

namespace epidemics {

ModelData::ModelData(
        std::vector<std::string> regionKeys_,
        std::vector<double> Ni_,
        std::vector<double> Mij_,
        std::vector<double> Cij_,
        std::vector<double> Ui_) :
    regionKeys(std::move(regionKeys_)),
    Ni(std::move(Ni_)),
    Mij(std::move(Mij_)),
    Cij(std::move(Cij_)),
    Ui(std::move(Ui_))
{
    init();
}

void ModelData::init() {
    size_t K = regionKeys.size();
    numRegions = K;
    invNi.resize(Ni.size(), 0.0);
    for (size_t i = 0; i < invNi.size(); ++i)
        invNi[i] = 1.0 / Ni[i];

    C_plus_Ct.resize(Cij.size());
    for (size_t i = 0; i < K; ++i)
    for (size_t j = 0; j < K; ++j)
        C_plus_Ct[i * K + j] = Cij[i * K + j] + Cij[j * K + i];

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
