import libepidemics.cantons.seiin_interventions as model

from scipy.integrate import solve_ivp
import numpy as np

from common import TestCaseEx, get_canton_model_data

# TODO: Test external commuters! Set days to > 0.

def py_solve(p, y0, t_eval, *, md):
    """Solve the SEIIN+interventions equation.

    Arguments:
        p: model parameters
        y0: (S0, E0, Ir0, Iu0, N0) initial condition at t=0
        t_eval: A list of times `t` to return the values of the ODE at.
        md: model data (design parameters)

    Returns:
        A list of 5-tuples, one tuple for each element of t_eval.
    """
    K = md.num_regions
    Mij = np.array(md.Mij).reshape((K, K))

    invD = 1 / p.D
    invZ = 1 / p.Z

    colsumMij = np.sum(Mij, axis=0)
    rowsumMij = np.sum(Mij, axis=1)

    def rhs(t, y):
        assert len(y) == 5 * K, (len(y), K)
        S  = y[0 * K : 1 * K]
        E  = y[1 * K : 2 * K]
        Ir = y[2 * K : 3 * K]
        Iu = y[3 * K : 4 * K]
        N  = y[4 * K : 5 * K]

        if t < p.d1:
            beta = p.beta
            theta = p.theta
        elif t < p.d2:
            beta = p.b1
            theta = p.theta1
        elif t < p.d3:
            beta = p.b2
            theta = p.theta2
        else:
            beta = p.b2 + (t - p.d3) * p.b3
            if beta > p.beta:
                beta = p.beta
            theta = p.theta3

        # Here we try to avoid repeating any computation...
        tmpI = beta * S / N * (Ir + p.mu * Iu)
        tmpE_Z = E * invZ
        tmpalphaE_Z = p.alpha * tmpE_Z

        dS = -tmpI
        dE = tmpI - tmpE_Z
        dIr = tmpalphaE_Z - Ir * invD
        dIu = tmpE_Z - tmpalphaE_Z - Iu * invD
        # dN = np.zeros(K)

        tmpNI = N - Ir
        tmpS_NI = S / tmpNI
        tmpE_NI = E / tmpNI
        tmpIu_NI = Iu / tmpNI
        dS  += theta * (Mij @ tmpS_NI  - colsumMij * tmpS_NI)
        dE  += theta * (Mij @ tmpE_NI  - colsumMij * tmpE_NI)
        dIu += theta * (Mij @ tmpIu_NI - colsumMij * tmpIu_NI)
        dN   = theta * (rowsumMij - colsumMij)  # This can be also set to 0.

        return [*dS, *dE, *dIr, *dIu, *dN]

    # Initial derivatives are 0.
    results = solve_ivp(rhs, y0=y0, t_span=(0, t_eval[-1]), t_eval=t_eval, rtol=1e-9, atol=1e-9, max_step=0.01)
    results = np.transpose(results.y)
    assert len(results) == len(t_eval)
    assert len(results[0]) == K * 5
    return results



class TestCantonsSEIINInterventions(TestCaseEx):
    def test_model_interventions(self):
        """Test the C++ implementation of the SEIIN model."""
        K = 3  # Number of cantons.
        md = get_canton_model_data(K=K, days=0)
        solver = model.Solver(md)

        # NOTE: The C++ and Python implementation do not produce exactly the
        # same results when interventions are used because the RHS is not
        # continuous. Lowering dt does not help much.
        params = model.Parameters(beta=0.3, mu=0.7, alpha=0.03, Z=4.0, D=5.0, theta=0.789,
                                  # b1=0.25, b2=0.21, b3=0.15,
                                  # d1=10, d2=18, d3=23,
                                  # theta1=0.5, theta2=0.3, theta3=0.2
                                  b1=0.3, b2=0.3, b3=0.3,
                                  d1=10, d2=18, d3=23,
                                  theta1=0.6, theta2=0.6, theta3=0.6)

        # S..., E..., Ir..., Iu..., N....
        y0 = (1.0e5, 0.9e5, 0.8e5, 1, 2, 3, 5, 6, 7, 0, 1, 2, 300000, 200000, 100000)
        t_eval = list(range(30))
        py_result = py_solve(params, y0=y0, t_eval=t_eval, md=md)
        y0 = model.State(y0)
        cpp_result = solver.solve(params, y0, t_eval=t_eval, dt=0.1)

        # Skip t=0 because relative error is undefined. Removing t=0 from t_eval does not work.
        for py, cpp in zip(py_result[1:], cpp_result[1:]):
            # See common.TestCaseEx.assertRelative
            for k in range(K):
                self.assertRelative(cpp.S(k),  py[0 * K + k], tolerance=1e-7)
                self.assertRelative(cpp.E(k),  py[1 * K + k], tolerance=1e-7)
                self.assertRelative(cpp.Ir(k), py[2 * K + k], tolerance=1e-7)
                self.assertRelative(cpp.Iu(k), py[3 * K + k], tolerance=1e-7)
                self.assertRelative(cpp.N(k),  py[4 * K + k], tolerance=1e-7)
