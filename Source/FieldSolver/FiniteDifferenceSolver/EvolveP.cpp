/*Evolution of the polarization P is governed by mu*d^2P/dt^2 + gamma*dP/dt = E_eff
 *Let dP/dt = v, then we need to solve the system of following two first-order ODEs
 *dP/dt = v
 *dv/dt = (E - gamma*v)/mu
 */


// Define the function representing the system of first-order ODEs
AMREX_GPU_DEVICE
void dPdt(const Real t, const Real* P, Real* dPdt_result, const Real mu, const Real gamma, const Real E)
{
    dPdt_result[0] = P[1];
    dPdt_result[1] = E / mu - gamma * P[1] / mu;
}

// Forward Euler time integrator
void forwardEuler(const Real t, MultiFab& P, const Real dt, const Real mu, const Real gamma, const Real E)
{
    Real k[2];

    for (MFIter mfi(P); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const auto& P_arr = P.array(mfi);

        // Calculate derivatives
        dPdt(t, &P_arr(bx.loVect(), 0), k, mu, gamma, E);

        // Update P using forward Euler method
        for (int i = 0; i < 2; ++i)
            P_arr(bx.loVect(), i) += dt * k[i];
    }
}
