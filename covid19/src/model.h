#pragma once

#include <cassert>
#include <vector>
#include <utility>

#include "data.h"
#include "integrator.h"

namespace epidemics {

/// State of the system.
struct State {
    /*
     * Using custom types with boost::odeint is not that simple. Instead, we
     * use a single std::vector<> to store the whole state of the simulation,
     * and the wrapper below to access the data in a human-friendly way.
     */
    static constexpr size_t kVarsPerRegion = 5;
    using RawState = std::vector<double>;

    /// Create a state with all values set to 0.
    explicit State(size_t numRegions) :
        numRegions_{numRegions},
        v_(kVarsPerRegion * numRegions, 0.0)
    { }

    /// Create a state from a raw state.
    explicit State(RawState state) :
        numRegions_{state.size() / kVarsPerRegion},
        v_{std::move(state)}
    {
        assert(v_.size() % kVarsPerRegion == 0);
    }

    size_t numRegions() const noexcept { return numRegions_; }
    RawState &raw() & { return v_; }
    const RawState &raw() const & { return v_; }
    RawState raw() && { return std::move(v_); }

    size_t size() noexcept { return v_.size(); }

    double &S (size_t i) { return this->v_[0 * this->numRegions_ + i]; }
    double &E (size_t i) { return this->v_[1 * this->numRegions_ + i]; }
    double &Ir(size_t i) { return this->v_[2 * this->numRegions_ + i]; }
    double &Iu(size_t i) { return this->v_[3 * this->numRegions_ + i]; }
    double &N (size_t i) { return this->v_[4 * this->numRegions_ + i]; }

    const double &S (size_t i) const { return this->v_[0 * this->numRegions_ + i]; }
    const double &E (size_t i) const { return this->v_[1 * this->numRegions_ + i]; }
    const double &Ir(size_t i) const { return this->v_[2 * this->numRegions_ + i]; }
    const double &Iu(size_t i) const { return this->v_[3 * this->numRegions_ + i]; }
    const double &N (size_t i) const { return this->v_[4 * this->numRegions_ + i]; }

private:
    size_t numRegions_;
    RawState v_;
};


/// Lightweight parameters (optimized for).
struct Parameters {
    static constexpr size_t numParameters = 15;

    double beta;   /// Transmission rate.
    double mu;     /// Reduction factor for transmission rate of undocumented individuals.
    double alpha;  /// Fraction of documented infections.
    double Z;      /// Average latency period.
    double D;      /// Average duration of infection.
    double theta;  /// Corrective multiplicative factor for Mij.

    // Intervention parameters.
    double b1;     /// beta after 1st intervention.
    double b2;     /// beta after 2nd intervention.
    double b3;     /// beta after 3rd intervention.
    double d1;     /// day of 1st intervention.
    double d2;     /// day of 2nd intervention.
    double d3;     /// day of 3rd intervention.
    double theta1; //theta after 1st intervention.
    double theta2; //theta after 2nd intervention.
    double theta3; //theta after 3rd intervention.
};


class Solver {
public:
    Solver(ModelData md) : md_{std::move(md)} { }

    // Data accessors.
    const ModelData &md() const noexcept { return md_; }

    size_t stateSize() const noexcept {
        return State::kVarsPerRegion * md_.numRegions;
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

    /// Solve the ODE.
    std::vector<State> solve(
            const Parameters &parameters,
            State y0,
            const std::vector<double> &tEval,
            IntegratorSettings settings) const;

private:
    /// Compute the RHS of the ODE.
    void _rhs(double t,
              Parameters p,
              const State & __restrict__ x,
              State & __restrict__ dxdt) const;

    ModelData md_;
};

}  // namespace epidemics
