#include "common.h"

#include <data.h>

namespace epidemics {
namespace cantons {

/// Helper function factory for State getters.
template <class State>
static auto makeValuesGetter(size_t valueIndex) {
    assert(0 <= valueIndex && valueIndex < State::kVarsPerRegion);
    /// Extract a subvector of the state corresponding to the given value.
    return [valueIndex](const State &state) {
        const double *p = state.raw().data();
        return std::vector<double>(p + valueIndex * state.numRegions(),
                                   p + (valueIndex + 1) * state.numRegions());
    };
}

}  // namespace cantons
}  // namespace epidemics
