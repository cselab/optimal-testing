#pragma once

#include "integrator.h"
#include "utils/signal.h"

#include <boost/numeric/odeint.hpp>

namespace epidemics {

template <typename RHS, typename State>
std::vector<State> integrate(
        RHS rhs,
        State y0,
        const std::vector<double> &tEval,
        IntegratorSettings settings)
{
    using RawState = typename State::RawState;
    using Stepper = boost::numeric::odeint::runge_kutta_dopri5<RawState>;

    std::vector<State> result;
    result.reserve(tEval.size());

    auto observer = [&result](const RawState &y, double /*t*/) mutable {
        if (check_signals_func)
            check_signals_func();
        result.push_back(State{y});
    };

    auto rhsWrapper = [rhs = std::move(rhs)](
            const RawState &x_, RawState &dxdt_, double t) {
        // This is a tricky part, we transform RawState to State during
        // computation and then at the end transform it back.
        State x{std::move(const_cast<RawState&>(x_))};
        State dxdt{std::move(dxdt_)};

        rhs(t, x, dxdt);

        /// Transform State back to RawState.
        const_cast<RawState &>(x_) = std::move(x).raw();
        dxdt_ = std::move(dxdt).raw();
    };

    typename State::RawState y0_(std::move(y0).raw());
    boost::numeric::odeint::integrate_times(
            Stepper{}, rhsWrapper, y0_,
            tEval.begin(), tEval.end(), settings.dt, observer);
    return result;
}

}  // namespace epidemics
