#include "MacroscopicProperties.H"
#include "WarpX.H"
#include "Utils/WarpXUtil.H"

#include <AMReX_ParmParse.H>

#include <memory>

using namespace amrex;

MacroscopicProperties::MacroscopicProperties ()
{
    ReadParameters();
}

void
MacroscopicProperties::ReadParameters ()
{
    ParmParse pp_macroscopic("macroscopic");
    // Since macroscopic maxwell solve is turned on,
    // user-defined sigma, mu, and epsilon are queried.
    // The vacuum values are used as default for the macroscopic parameters
    // with a warning message to the user to indicate that no value was specified.

    // Query input for material conductivity, sigma.
    bool sigma_specified = false;
    if (queryWithParser(pp_macroscopic, "sigma", m_sigma)) {
        m_sigma_s = "constant";
        sigma_specified = true;
    }
    if (pp_macroscopic.query("sigma_function(x,y,z)", m_str_sigma_function) ) {
        m_sigma_s = "parse_sigma_function";
        sigma_specified = true;
    }
    if (!sigma_specified) {
        amrex::Print() << "WARNING: Material conductivity is not specified. Using default vacuum value of " << m_sigma << " in the simulation\n";
    }
    // initialization of sigma (conductivity) with parser
    if (m_sigma_s == "parse_sigma_function") {
        Store_parserString(pp_macroscopic, "sigma_function(x,y,z)", m_str_sigma_function);
        m_sigma_parser = std::make_unique<ParserWrapper<3>>(
                                 makeParser(m_str_sigma_function,{"x","y","z"}));
    }

    bool epsilon_specified = false;
    if (queryWithParser(pp_macroscopic, "epsilon", m_epsilon)) {
        m_epsilon_s = "constant";
        epsilon_specified = true;
    }
    if (pp_macroscopic.query("epsilon_function(x,y,z)", m_str_epsilon_function) ) {
        m_epsilon_s = "parse_epsilon_function";
        epsilon_specified = true;
    }
    if (!epsilon_specified) {
        amrex::Print() << "WARNING: Material permittivity is not specified. Using default vacuum value of " << m_epsilon << " in the simulation\n";
    }

    // initialization of epsilon (permittivity) with parser
    if (m_epsilon_s == "parse_epsilon_function") {
        Store_parserString(pp_macroscopic, "epsilon_function(x,y,z)", m_str_epsilon_function);
        m_epsilon_parser = std::make_unique<ParserWrapper<3>>(
                                 makeParser(m_str_epsilon_function,{"x","y","z"}));
    }

    // Query input for material permeability, mu
    bool mu_specified = false;
    if (queryWithParser(pp_macroscopic, "mu", m_mu)) {
        m_mu_s = "constant";
        mu_specified = true;
    }
    if (pp_macroscopic.query("mu_function(x,y,z)", m_str_mu_function) ) {
        m_mu_s = "parse_mu_function";
        mu_specified = true;
    }
    if (!mu_specified) {
        amrex::Print() << "WARNING: Material permeability is not specified. Using default vacuum value of " << m_mu << " in the simulation\n";
    }

    // initialization of mu (permeability) with parser
    if (m_mu_s == "parse_mu_function") {
        Store_parserString(pp_macroscopic, "mu_function(x,y,z)", m_str_mu_function);
        m_mu_parser = std::make_unique<ParserWrapper<3>>(
                                 makeParser(m_str_mu_function,{"x","y","z"}));
    }

#ifdef WARPX_MAG_LLG
    pp_macroscopic.get("mag_Ms_init_style", m_mag_Ms_s);
    if (m_mag_Ms_s == "constant") pp_macroscopic.get("mag_Ms", m_mag_Ms);
    // _mag_ such that it's clear the Ms variable is only meaningful for magnetic materials
    //initialization with parser
    if (m_mag_Ms_s == "parse_mag_Ms_function") {
        Store_parserString(pp_macroscopic, "mag_Ms_function(x,y,z)", m_str_mag_Ms_function);
        m_mag_Ms_parser.reset(new ParserWrapper<3>(
                                  makeParser(m_str_mag_Ms_function,{"x","y","z"})));
    }

    pp_macroscopic.get("mag_alpha_init_style", m_mag_alpha_s);
    if (m_mag_alpha_s == "constant") pp_macroscopic.get("mag_alpha", m_mag_alpha);
    // _mag_ such that it's clear the alpha variable is only meaningful for magnetic materials
    //initialization with parser
    if (m_mag_alpha_s == "parse_mag_alpha_function") {
        Store_parserString(pp_macroscopic, "mag_alpha_function(x,y,z)", m_str_mag_alpha_function);
        m_mag_alpha_parser.reset(new ParserWrapper<3>(
                                  makeParser(m_str_mag_alpha_function,{"x","y","z"})));
    }

    pp_macroscopic.get("mag_gamma_init_style", m_mag_gamma_s);
    if (m_mag_gamma_s == "constant") pp_macroscopic.get("mag_gamma", m_mag_gamma);
    // _mag_ such that it's clear the gamma variable parsed here is only meaningful for magnetic materials
    //initialization with parser
    if (m_mag_gamma_s == "parse_mag_gamma_function") {
        Store_parserString(pp_macroscopic, "mag_gamma_function(x,y,z)", m_str_mag_gamma_function);
        m_mag_gamma_parser.reset(new ParserWrapper<3>(
                                  makeParser(m_str_mag_gamma_function,{"x","y","z"})));
    }

    m_mag_normalized_error = 0.1;
    pp_macroscopic.query("mag_normalized_error",m_mag_normalized_error);

    m_mag_max_iter = 100;
    pp_macroscopic.query("mag_max_iter",m_mag_max_iter);

    m_mag_tol = 0.0001;
    pp_macroscopic.query("mag_tol",m_mag_tol);

#endif
}

void
MacroscopicProperties::InitData ()
{
    amrex::Print() << "we are in init data of macro \n";
    auto & warpx = WarpX::GetInstance();

    // Get BoxArray and DistributionMap of warpx instant.
    int lev = 0;
    BoxArray ba = warpx.boxArray(lev);
    DistributionMapping dmap = warpx.DistributionMap(lev);
    int ng = 3;
    // Define material property multifabs using ba and dmap from WarpX instance
    // sigma is cell-centered MultiFab
    m_sigma_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
    // epsilon is cell-centered MultiFab
    m_eps_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
    // mu is cell-centered MultiFab
    m_mu_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);

#ifdef WARPX_MAG_LLG
    // all magnetic macroparameters are stored on cell centers
    m_mag_Ms_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
    m_mag_alpha_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
    m_mag_gamma_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
#endif

    // Initialize sigma
    if (m_sigma_s == "constant") {

        m_sigma_mf->setVal(m_sigma);

    } else if (m_sigma_s == "parse_sigma_function") {

        InitializeMacroMultiFabUsingParser(m_sigma_mf.get(), getParser(m_sigma_parser), lev);
    }
    // Initialize epsilon
    if (m_epsilon_s == "constant") {

        m_eps_mf->setVal(m_epsilon);

    } else if (m_epsilon_s == "parse_epsilon_function") {

        InitializeMacroMultiFabUsingParser(m_eps_mf.get(), getParser(m_epsilon_parser), lev);

    }
    // Initialize mu
    if (m_mu_s == "constant") {

        m_mu_mf->setVal(m_mu);

    } else if (m_mu_s == "parse_mu_function") {

        InitializeMacroMultiFabUsingParser(m_mu_mf.get(), getParser(m_mu_parser), lev);

    }

#ifdef WARPX_MAG_LLG
    // mag_Ms - defined at cell centers
    if (m_mag_Ms_s == "constant") {
        m_mag_Ms_mf->setVal(m_mag_Ms);
    }
    else if (m_mag_Ms_s == "parse_mag_Ms_function"){
        InitializeMacroMultiFabUsingParser(m_mag_Ms_mf.get(), getParser(m_mag_Ms_parser), lev);
    }
    // if there are regions with Ms=0, the user must provide mur value there
    if (m_mag_Ms_mf->min(0,m_mag_Ms_mf->nGrow()) < 0._rt){
        amrex::Abort("Ms must be non-negative values");
    }
    else if (m_mag_Ms_mf->min(0,m_mag_Ms_mf->nGrow()) == 0._rt){
        if (m_mu_s != "constant" && m_mu_s != "parse_mu_function"){
            amrex::Abort("permeability must be specified since part of the simulation domain is non-magnetic !");
        }
    }

    // mag_alpha - defined at cell centers
    if (m_mag_alpha_s == "constant") {
        m_mag_alpha_mf->setVal(m_mag_alpha);
    }
    else if (m_mag_alpha_s == "parse_mag_alpha_function"){
        InitializeMacroMultiFabUsingParser(m_mag_alpha_mf.get(), getParser(m_mag_alpha_parser), lev);
    }
    if (m_mag_alpha_mf->min(0,m_mag_alpha_mf->nGrow()) < 0._rt) {
        amrex::Abort("alpha should be positive, but the user input has negative values");
    }

    // mag_gamma - defined at cell centers
    if (m_mag_gamma_s == "constant") {
        m_mag_gamma_mf->setVal(m_mag_gamma);

    }
    else if (m_mag_gamma_s == "parse_mag_gamma_function"){
        InitializeMacroMultiFabUsingParser(m_mag_gamma_mf.get(), getParser(m_mag_gamma_parser), lev);
    }
    if (m_mag_gamma_mf->max(0,m_mag_gamma_mf->nGrow()) > 0._rt) {
        amrex::Abort("gamma should be negative, but the user input has positive values");
    }
#endif

    IntVect sigma_stag = m_sigma_mf->ixType().toIntVect();
    IntVect epsilon_stag = m_eps_mf->ixType().toIntVect();
    IntVect mu_stag = m_mu_mf->ixType().toIntVect();
    IntVect Ex_stag = warpx.getEfield_fp(0,0).ixType().toIntVect();
    IntVect Ey_stag = warpx.getEfield_fp(0,1).ixType().toIntVect();
    IntVect Ez_stag = warpx.getEfield_fp(0,2).ixType().toIntVect();
#ifdef WARPX_MAG_LLG
    IntVect mag_Ms_stag = m_mag_Ms_mf->ixType().toIntVect(); //cell-centered
    IntVect mag_alpha_stag = m_mag_alpha_mf->ixType().toIntVect();
    IntVect mag_gamma_stag = m_mag_gamma_mf->ixType().toIntVect();
    IntVect Mx_stag = warpx.getMfield_fp(0,0).ixType().toIntVect(); // face-centered
    IntVect My_stag = warpx.getMfield_fp(0,1).ixType().toIntVect();
    IntVect Mz_stag = warpx.getMfield_fp(0,2).ixType().toIntVect();
#endif
    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        sigma_IndexType[idim]   = sigma_stag[idim];
        epsilon_IndexType[idim] = epsilon_stag[idim];
        mu_IndexType[idim]      = mu_stag[idim];
        Ex_IndexType[idim]      = Ex_stag[idim];
        Ey_IndexType[idim]      = Ey_stag[idim];
        Ez_IndexType[idim]      = Ez_stag[idim];
#ifdef WARPX_MAG_LLG
        mag_Ms_IndexType[idim]    = mag_Ms_stag[idim];
        mag_alpha_IndexType[idim] = mag_alpha_stag[idim];
        mag_gamma_IndexType[idim] = mag_gamma_stag[idim];
        Mx_IndexType[idim]        = Mx_stag[idim];
        My_IndexType[idim]        = My_stag[idim];
        Mz_IndexType[idim]        = Mz_stag[idim];
#endif
        macro_cr_ratio[idim]    = 1;
    }
#if (AMREX_SPACEDIM==2)
        sigma_IndexType[2]   = 0;
        epsilon_IndexType[2] = 0;
        mu_IndexType[2]      = 0;
        Ex_IndexType[2]      = 0;
        Ey_IndexType[2]      = 0;
        Ez_IndexType[2]      = 0;
#ifdef WARPX_MAG_LLG
        mag_Ms_IndexType[2]    = 0;
        mag_alpha_IndexType[2] = 0;
        mag_gamma_IndexType[2] = 0;
        Mx_IndexType[2]        = 0;
        My_IndexType[2]        = 0;
        Mz_IndexType[2]        = 0;
#endif
        macro_cr_ratio[2]    = 1;
#endif


}

void
MacroscopicProperties::InitializeMacroMultiFabUsingParser (
                       MultiFab *macro_mf, HostDeviceParser<3> const& macro_parser,
                       int lev)
{
    auto& warpx = WarpX::GetInstance();
    const auto dx_lev = warpx.Geom(lev).CellSizeArray();
    const RealBox& real_box = warpx.Geom(lev).ProbDomain();
    IntVect iv = macro_mf->ixType().toIntVect();
    IntVect grown_iv = macro_mf->nGrowVect();
    for ( MFIter mfi(*macro_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        // Initialize ghost cells in addition to valid cells

        const Box& tb = mfi.growntilebox(grown_iv);
        auto const& macro_fab =  macro_mf->array(mfi);
        amrex::ParallelFor (tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Shift x, y, z position based on index type
                Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                amrex::Real y = 0._rt;
                Real fac_z = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                Real z = j * dx_lev[1] + real_box.lo(1) + fac_z;
#else
                Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;
                Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
                Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                // initialize the macroparameter
                macro_fab(i,j,k) = macro_parser(x,y,z);
        });

    }
}
