#include "data.h"
#include <utils/assert.h>

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


    fclose(f);

    return out;
}


}  // namespace epidemics
