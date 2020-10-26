#pragma once

#include "base.h"

namespace epidemics {
namespace cantons {
namespace seiin_interventions {

/// State of the system.
struct State : StateBase<5> {
    using StateBase<5>::StateBase;

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

struct Solver : SolverBase<Solver, State, Parameters> {
    using SolverBase<Solver, State, Parameters>::SolverBase;

    void rhs(double t,
             Parameters p,
             const State & __restrict__ x,
             State & __restrict__ dxdt) const
    {
        int day = static_cast<int>(t);
        for (size_t i = 0; i < md_.numRegions; ++i) {
            double extComIu = md_.getExternalCommutersIu(day, i);

            // Interventions: beta is modelled as a function of time.
            double BETA;
            double THETA;
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

            double A = BETA * x.S(i) / x.N(i) * (x.Ir(i) + extComIu);
            double B = BETA * x.S(i) / x.N(i) * p.mu * x.Iu(i);
            double E_Z = x.E(i) / p.Z;

            double dS = -(A + B);
            double dE = A + B - E_Z;
            double dIr = p.alpha * E_Z - x.Ir(i) / p.D;
            double dIu = E_Z - p.alpha * E_Z - x.Iu(i) / p.D;
            double dN = 0.0;

            double inv = 1 / (x.N(i) - x.Ir(i));
            //for (size_t j = 0; j < md_.numRegions; ++j) { // XXX
            for (size_t j : this->nonzero_Mij(i)) {
                double Tij = this->M(i, j) / (x.N(j) - x.Ir(j));
                double Tji = this->M(j, i) * inv;
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
