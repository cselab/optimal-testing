#pragma once

#include <vector>

namespace epidemics {

struct IntegratorSettings {
    double dt{0.1};
};

template <typename RHS, typename State>
std::vector<State> integrate(
        RHS rhs,
        State y0,
        const std::vector<double> &tEval,
        IntegratorSettings settings);

}  // namespace epidemics
