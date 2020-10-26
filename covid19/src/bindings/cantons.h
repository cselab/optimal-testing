#include "common.h"

#include <data.h>

namespace epidemics {
namespace cantons {

/// Helper function factory for State getters.
template <template <typename> class State, typename T>
static auto makeValuesGetter(size_t valueIndex) {
    assert(0 <= valueIndex && valueIndex < State<T>::kVarsPerRegion);
    /// Extract a subvector of the state corresponding to the given value.
    return [valueIndex](const State<T> &state) {
        const T *p = state.raw().data();
        return std::vector<T>(p + valueIndex * state.numRegions(),
                              p + (valueIndex + 1) * state.numRegions());
    };
}

}  // namespace cantons
}  // namespace epidemics
