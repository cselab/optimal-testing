#include "data.h"
#include <utils/assert.h>

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


ModelData readModelData(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (f == nullptr)
        DIE("Error opening file \"%s\". Did you forget to run ./py/data.py?\n", filename);

    int N;  // Number of regions.
    if (fscanf(f, "%d", &N) != 1)
        DIE("Reading number of regions failed.");

    ModelData out;
    out.regionKeys.resize(N);
    for (int i = 0; i < N; ++i) {
        char name[64];
        if (fscanf(f, "%s", name) != 1)
            DIE("Reading name of the region #%d failed.\n", i);
        out.regionKeys[i] = name;
    }

    out.Ni.resize(N);
    for (double &pop : out.Ni)
        if (fscanf(f, "%lg", &pop) != 1)
            DIE("Reading region population failed.\n");

    out.Mij.resize(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if (fscanf(f, "%lf", &out.Mij[i * N + j]) != 1)
                DIE("Reading Mij[%d][%d] failed.\n", i, j);
            if (i == j)
                out.Mij[i * N + j] = 0.0;
        }

    out.Ui.resize(N);
    for (double &u : out.Ui)
        if (fscanf(f, "%lg", &u) != 1)
            DIE("Reading user-defined failed.\n");

    fclose(f);

    return out;
}


}  // namespace epidemics
