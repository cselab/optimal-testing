#pragma once

#include "data.h"
#include <integrator.h>

#include <cassert>

namespace epidemics {
namespace cantons {

/*
 * Using custom types with boost::odeint is not that simple. Instead, we use a
 * single std::vector<> to store the whole state of the simulation, and the
 * wrapper below to access the data in a human-friendly way.
 */
template <typename T, size_t VarsPerRegion>
struct StateBase {
    using RawState = std::vector<T>;
    static constexpr size_t kVarsPerRegion = VarsPerRegion;

    /// Create a state with all values set to 0.
    explicit StateBase(size_t numRegions) :
        numRegions_{numRegions},
        v_(kVarsPerRegion * numRegions, 0.0)
    { }

    /// Create a state from a raw state.
    explicit StateBase(RawState state) :
        numRegions_{state.size() / kVarsPerRegion},
        v_{std::move(state)}
    {
        assert(v_.size() % kVarsPerRegion == 0);
    }

    size_t numRegions() const noexcept { return numRegions_; }
    RawState &raw() & { return v_; }
    const RawState &raw() const & { return v_; }
    RawState raw() && { return std::move(v_); }

    constexpr size_t size() noexcept { return v_.size(); }

protected:
    size_t numRegions_;
    RawState v_;
};


/** CRTP base class for solvers.
 *
 * Solvers have to only define a `rhs` function, the integrator is handled by
 * the base class in `base.hh`.
 */
template <typename Derived,
          template <typename> class State,
          template <typename> class Parameters>
class SolverBase {
public:
    SolverBase(ModelData md) :
        md_{std::move(md)}
    { }

    const ModelData &md() const noexcept { return md_; }

    size_t stateSize() const noexcept {
        return State<double>::kVarsPerRegion * md_.numRegions;
    }

    double M(int from, int to) const {
        return md_.Mij[from * md_.numRegions + to];
    }
    const std::vector<size_t>& nonzero_Mij(size_t from_or_in) const {
        return md_.nonzero_Mij[from_or_in];
    }
    double C(int from, int to) const {
        return md_.Cij[from * md_.numRegions + to];
    }
    double C_plus_Ct(int from, int to) const {
        return md_.C_plus_Ct[from * md_.numRegions + to];
    }

    template <typename T>
    std::vector<State<T>> solve(
            const Parameters<T> &parameters,
            State<T> y0,
            const std::vector<double> &tEval,
            IntegratorSettings settings) const
    {
        if (y0.raw().size() != md_.numRegions * State<T>::kVarsPerRegion)
            throw std::invalid_argument("Invalid state vector length.");

        return integrate(
                [this, parameters](double t, const State<T> &x, State<T> &dxdt) {
                    assert(x.raw().size() == dxdt.raw().size());
                    assert(x.raw().size() == md_.numRegions * State<T>::kVarsPerRegion);
                    return derived()->rhs(t, parameters, x, dxdt);
                },
                std::move(y0), tEval, std::move(settings));
    }

protected:
    Derived *derived() noexcept {
        return static_cast<Derived *>(this);
    }
    const Derived *derived() const noexcept {
        return static_cast<const Derived *>(this);
    }

    ModelData md_;
};

}  // namespace cantons
}  // namespace epidemics
