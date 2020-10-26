#pragma once

#include "base.h"

namespace epidemics {
namespace cantons {
namespace seiin_interventions {

/// State of the system.
template <typename T>
struct State : StateBase<T, 5> {
    using StateBase<T, 5>::StateBase;
    T &S (size_t i) { return this->v_[0 * this->numRegions_ + i]; }
    T &E (size_t i) { return this->v_[1 * this->numRegions_ + i]; }
    T &Ir(size_t i) { return this->v_[2 * this->numRegions_ + i]; }
    T &Iu(size_t i) { return this->v_[3 * this->numRegions_ + i]; }
    T &N (size_t i) { return this->v_[4 * this->numRegions_ + i]; }

    const T &S (size_t i) const { return this->v_[0 * this->numRegions_ + i]; }
    const T &E (size_t i) const { return this->v_[1 * this->numRegions_ + i]; }
    const T &Ir(size_t i) const { return this->v_[2 * this->numRegions_ + i]; }
    const T &Iu(size_t i) const { return this->v_[3 * this->numRegions_ + i]; }
    const T &N (size_t i) const { return this->v_[4 * this->numRegions_ + i]; }
};


/// Lightweight parameters (optimized for).
template <typename T>
struct Parameters {
    static constexpr size_t numParameters = 15;

    T beta;   /// Transmission rate.
    T mu;     /// Reduction factor for transmission rate of undocumented individuals.
    T alpha;  /// Fraction of documented infections.
    T Z;      /// Average latency period.
    T D;      /// Average duration of infection.
    T theta;  /// Corrective multiplicative factor for Mij.

    // Intervention parameters.
    T b1;     /// beta after 1st intervention.
    T b2;     /// beta after 2nd intervention.
    T b3;     /// beta after 3rd intervention.
    T d1;     /// day of 1st intervention.
    T d2;     /// day of 2nd intervention.
    T d3;     /// day of 3rd intervention.
    T theta1; //theta after 1st intervention.
    T theta2; //theta after 2nd intervention.
    T theta3; //theta after 3rd intervention.
};

struct Solver : SolverBase<Solver, State, Parameters> {
    using SolverBase<Solver, State, Parameters>::SolverBase;

    template <typename T>
    void rhs(double t,
             Parameters<T> p,
             const State<T> & __restrict__ x,
             State<T> & __restrict__ dxdt) const
    {
        const T ZERO = 0 * p.beta;
        int day = static_cast<int>(t);
        for (size_t i = 0; i < md_.numRegions; ++i) {
            double extComIu = md_.getExternalCommutersIu(day, i);

            // Interventions: beta is modelled as a function of time.
            T BETA;
            T THETA;
            if ( day < p.d1) {
               BETA = p.beta;
               THETA = p.theta;
            } else if (day < p.d2) {
               BETA = p.b1;
               THETA = p.theta1;
            } else if (day < p.d3) {
               BETA = p.b2;
               THETA = p.theta2;
            } else {
               BETA = p.b2 + (day - p.d3) * p.b3;
               if (BETA > p.beta)
                  BETA = p.beta;
               THETA = p.theta3;
            }

            T A = BETA * x.S(i) / x.N(i) * (x.Ir(i) + extComIu);
            T B = BETA * x.S(i) / x.N(i) * p.mu * x.Iu(i);
            T E_Z = x.E(i) / p.Z;

            T dS = -(A + B);
            T dE = A + B - E_Z;
            T dIr = p.alpha * E_Z - x.Ir(i) / p.D;
            T dIu = E_Z - p.alpha * E_Z - x.Iu(i) / p.D;
            T dN = ZERO;

            T inv = 1 / (x.N(i) - x.Ir(i));
            //for (size_t j = 0; j < md_.numRegions; ++j) { // XXX
            for (size_t j : this->nonzero_Mij(i)) {
                T Tij = this->M(i, j) / (x.N(j) - x.Ir(j));
                T Tji = this->M(j, i) * inv;
                dS += THETA * (Tij * x.S(j) - Tji * x.S(i));
                dE += THETA * (Tij * x.E(j) - Tji * x.E(i));
                // Documented infected people are in quarantine, they do not move around.
                // dIr += p.theta * (Tij * x.Ir(j) - Tji * x.Ir(i));
                dIu += THETA * (Tij * x.Iu(j) - Tji * x.Iu(i));
                dN  += THETA * (this->M(i, j) - this->M(j, i));
            }

            dxdt.S(i) = dS;
            dxdt.E(i) = dE;
            dxdt.Ir(i) = dIr;
            dxdt.Iu(i) = dIu;
            dxdt.N(i) = dN;
        }
    }
};

}  // namespace seiin_interventions
}  // namespace cantons
}  // namespace epidemics
