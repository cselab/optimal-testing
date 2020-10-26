#include "model.h"
#include "integrator.hh"

namespace epidemics {

void Solver::_rhs(double t,
                  Parameters p,
                  const State & __restrict__ x,
                  State & __restrict__ dxdt) const
{
    int day = static_cast<int>(t);
    for (size_t i = 0; i < md_.numRegions; ++i) {

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

        double A = BETA * x.S(i) / x.N(i) * x.Ir(i);
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

std::vector<State> Solver::solve(
        const Parameters &parameters,
        State y0,
        const std::vector<double> &tEval,
        IntegratorSettings settings) const
{
    if (y0.raw().size() != md_.numRegions * State::kVarsPerRegion)
        throw std::invalid_argument("Invalid state vector length.");

    return integrate(
            [this, parameters](double t, const State &x, State &dxdt) {
                assert(x.raw().size() == dxdt.raw().size());
                assert(x.raw().size() == md_.numRegions * State::kVarsPerRegion);
                return this->_rhs(t, parameters, x, dxdt);
            },
            std::move(y0), tEval, std::move(settings));
}

}  // namespace epidemics
