#include "MacroscopicProperties.H"

#include "Utils/WarpXUtil.H"
#include "WarpX.H"

#include <AMReX_Array4.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>

#include <AMReX_BaseFwd.H>

#include <memory>
#include <sstream>

using namespace amrex;

GetSigmaMacroparameter::GetSigmaMacroparameter () noexcept
{
    auto& warpx = WarpX::GetInstance();
    auto& macroscopic_properties = warpx.GetMacroscopicProperties();
    if (macroscopic_properties.m_sigma_s == "constant") {
        m_type = ConstantValue;
        m_value = macroscopic_properties.m_sigma;
    }
    else if (macroscopic_properties.m_sigma_s == "parse_sigma_function") {
        m_type = ParserFunction;
        m_parser = macroscopic_properties.m_sigma_parser->compile<3>();
    }
}

GetMuMacroparameter::GetMuMacroparameter () noexcept
{
    auto& warpx = WarpX::GetInstance();
    auto& macroscopic_properties = warpx.GetMacroscopicProperties();
    if (macroscopic_properties.m_mu_s == "constant") {
        m_type = ConstantValue;
        m_value = macroscopic_properties.m_mu;
    }
    else if (macroscopic_properties.m_mu_s == "parse_mu_function") {
        m_type = ParserFunction;
        m_parser = macroscopic_properties.m_mu_parser->compile<3>();
    }
}

GetEpsilonMacroparameter::GetEpsilonMacroparameter () noexcept
{
    auto& warpx = WarpX::GetInstance();
    auto& macroscopic_properties = warpx.GetMacroscopicProperties();
    if (macroscopic_properties.m_epsilon_s == "constant") {
        m_type = ConstantValue;
        m_value = macroscopic_properties.m_epsilon;
    }
    else if (macroscopic_properties.m_epsilon_s == "parse_epsilon_function") {
        m_type = ParserFunction;
        m_parser = macroscopic_properties.m_epsilon_parser->compile<3>();
    }
}


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
        std::stringstream warnMsg;
        warnMsg << "Material conductivity is not specified. Using default vacuum value of " <<
            m_sigma << " in the simulation.";
        WarpX::GetInstance().RecordWarning("Macroscopic properties",
            warnMsg.str());
    }
    // initialization of sigma (conductivity) with parser
    if (m_sigma_s == "parse_sigma_function") {
        Store_parserString(pp_macroscopic, "sigma_function(x,y,z)", m_str_sigma_function);
        m_sigma_parser = std::make_unique<Parser>(
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
        std::stringstream warnMsg;
        warnMsg << "Material permittivity is not specified. Using default vacuum value of " <<
            m_epsilon << " in the simulation.";
        WarpX::GetInstance().RecordWarning("Macroscopic properties",
            warnMsg.str());
    }

    // initialization of epsilon (permittivity) with parser
    if (m_epsilon_s == "parse_epsilon_function") {
        Store_parserString(pp_macroscopic, "epsilon_function(x,y,z)", m_str_epsilon_function);
        m_epsilon_parser = std::make_unique<Parser>(
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
        std::stringstream warnMsg;
        warnMsg << "Material permittivity is not specified. Using default vacuum value of " <<
            m_mu << " in the simulation.";
        WarpX::GetInstance().RecordWarning("Macroscopic properties",
            warnMsg.str());
    }

    // initialization of mu (permeability) with parser
    if (m_mu_s == "parse_mu_function") {
        Store_parserString(pp_macroscopic, "mu_function(x,y,z)", m_str_mu_function);
        m_mu_parser = std::make_unique<Parser>(
                                 makeParser(m_str_mu_function,{"x","y","z"}));
    }

#ifdef WARPX_MAG_LLG
    auto &warpx = WarpX::GetInstance();
    pp_macroscopic.get("mag_Ms_init_style", m_mag_Ms_s);
    if (m_mag_Ms_s == "constant") pp_macroscopic.get("mag_Ms", m_mag_Ms);
    // _mag_ such that it's clear the Ms variable is only meaningful for magnetic materials
    //initialization with parser
    if (m_mag_Ms_s == "parse_mag_Ms_function") {
        Store_parserString(pp_macroscopic, "mag_Ms_function(x,y,z)", m_str_mag_Ms_function);
        m_mag_Ms_parser = std::make_unique<amrex::Parser>(
                                  makeParser(m_str_mag_Ms_function,{"x","y","z"}));
    }

    pp_macroscopic.get("mag_alpha_init_style", m_mag_alpha_s);
    if (m_mag_alpha_s == "constant") pp_macroscopic.get("mag_alpha", m_mag_alpha);
    // _mag_ such that it's clear the alpha variable is only meaningful for magnetic materials
    //initialization with parser
    if (m_mag_alpha_s == "parse_mag_alpha_function") {
        Store_parserString(pp_macroscopic, "mag_alpha_function(x,y,z)", m_str_mag_alpha_function);
        m_mag_alpha_parser = std::make_unique<amrex::Parser>(
                                  makeParser(m_str_mag_alpha_function,{"x","y","z"}));
    }

    pp_macroscopic.get("mag_gamma_init_style", m_mag_gamma_s);
    if (m_mag_gamma_s == "constant") pp_macroscopic.get("mag_gamma", m_mag_gamma);
    // _mag_ such that it's clear the gamma variable parsed here is only meaningful for magnetic materials
    //initialization with parser
    if (m_mag_gamma_s == "parse_mag_gamma_function") {
        Store_parserString(pp_macroscopic, "mag_gamma_function(x,y,z)", m_str_mag_gamma_function);
        m_mag_gamma_parser = std::make_unique<amrex::Parser>(
                                  makeParser(m_str_mag_gamma_function,{"x","y","z"}));
    }

    if (warpx.mag_LLG_exchange_coupling == 1) { // spin exchange coupling turned off by default
        pp_macroscopic.get("mag_exchange_init_style", m_mag_exchange_s);
        if (m_mag_exchange_s == "constant") pp_macroscopic.get("mag_exchange", m_mag_exchange);
        // _mag_ such that it's clear the exch variable is only meaningful for magnetic materials
        //initialization with parser
        if (m_mag_exchange_s == "parse_mag_exchange_function") {
            Store_parserString(pp_macroscopic, "mag_exchange_function(x,y,z)", m_str_mag_exchange_function);
            m_mag_exchange_parser = std::make_unique<amrex::Parser>(
                                      makeParser(m_str_mag_exchange_function,{"x","y","z"}));
        }
    }

    if (warpx.mag_LLG_anisotropy_coupling == 1) { // magnetic crystal is considered as isotropic by default
        pp_macroscopic.get("mag_anisotropy_init_style", m_mag_anisotropy_s);
        if (m_mag_anisotropy_s == "constant") pp_macroscopic.get("mag_anisotropy", m_mag_anisotropy);
        // _mag_ such that it's clear the exch variable is only meaningful for magnetic materials
        //initialization with parser
        if (m_mag_anisotropy_s == "parse_mag_anisotropy_function") {
            Store_parserString(pp_macroscopic, "mag_anisotropy_function(x,y,z)", m_str_mag_anisotropy_function);
            m_mag_anisotropy_parser = std::make_unique<amrex::Parser>(
                                      makeParser(m_str_mag_anisotropy_function,{"x","y","z"}));
        }
    }

    m_mag_normalized_error = 0.1;
    pp_macroscopic.query("mag_normalized_error",m_mag_normalized_error);

    m_mag_max_iter = 100;
    pp_macroscopic.query("mag_max_iter",m_mag_max_iter);

    m_mag_tol = 0.0001;
    pp_macroscopic.query("mag_tol",m_mag_tol);

    if (warpx.mag_LLG_anisotropy_coupling == 1) {
        amrex::Vector<amrex::Real> mag_LLG_anisotropy_axis_parser(3,0.0);
        // The anisotropy_axis for the anisotropy coupling term H_anisotropy in H_eff
        pp_macroscopic.getarr("mag_LLG_anisotropy_axis", mag_LLG_anisotropy_axis_parser);
        for (int i = 0; i < 3; i++) {
            mag_LLG_anisotropy_axis[i] = mag_LLG_anisotropy_axis_parser[i];
        }
    }

#endif
}

void
MacroscopicProperties::InitData ()
{
    amrex::Print() << "we are in init data of macro \n";
    auto & warpx = WarpX::GetInstance();

#ifdef WARPX_MAG_LLG
    // Get BoxArray and DistributionMap of warpx instant.
    int lev = 0;
    BoxArray ba = warpx.boxArray(lev);
    DistributionMapping dmap = warpx.DistributionMap(lev);
    const amrex::IntVect ng = warpx.getngE();

    // all magnetic macroparameters are stored on cell centers
    m_mag_Ms_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
    m_mag_alpha_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
    m_mag_gamma_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
    m_mag_exchange_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
    m_mag_anisotropy_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
    // mag_Ms - defined at cell centers
    if (m_mag_Ms_s == "constant") {
        m_mag_Ms_mf->setVal(m_mag_Ms);
    }
    else if (m_mag_Ms_s == "parse_mag_Ms_function"){
        InitializeMacroMultiFabUsingParser(m_mag_Ms_mf.get(), m_mag_Ms_parser->compile<3>(), lev);
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
        InitializeMacroMultiFabUsingParser(m_mag_alpha_mf.get(), m_mag_alpha_parser->compile<3>(), lev);
    }
    if (m_mag_alpha_mf->min(0,m_mag_alpha_mf->nGrow()) < 0._rt) {
        amrex::Abort("alpha should be positive, but the user input has negative values");
    }

    // mag_gamma - defined at cell centers
    if (m_mag_gamma_s == "constant") {
        m_mag_gamma_mf->setVal(m_mag_gamma);

    }
    else if (m_mag_gamma_s == "parse_mag_gamma_function"){
        InitializeMacroMultiFabUsingParser(m_mag_gamma_mf.get(), m_mag_gamma_parser->compile<3>(), lev);
    }
    if (m_mag_gamma_mf->max(0,m_mag_gamma_mf->nGrow()) > 0._rt) {
        amrex::Abort("gamma should be negative, but the user input has positive values");
    }

    // mag_exchange - defined at cell centers
    if (m_mag_exchange_s == "constant") {
        m_mag_exchange_mf->setVal(m_mag_exchange);
    }
    else if (m_mag_exchange_s == "parse_mag_exchange_function"){
        InitializeMacroMultiFabUsingParser(m_mag_exchange_mf.get(), m_mag_exchange_parser->compile<3>(), lev);
    }

    // mag_anisotropy - defined at cell centers
    if (m_mag_anisotropy_s == "constant") {
        m_mag_anisotropy_mf->setVal(m_mag_anisotropy);
    }
    else if (m_mag_anisotropy_s == "parse_mag_anisotropy_function"){
        InitializeMacroMultiFabUsingParser(m_mag_anisotropy_mf.get(),
                                           m_mag_anisotropy_parser->compile<3>(), lev);
    }
#endif


    IntVect Ex_stag = warpx.getEfield_fp(0,0).ixType().toIntVect();
    IntVect Ey_stag = warpx.getEfield_fp(0,1).ixType().toIntVect();
    IntVect Ez_stag = warpx.getEfield_fp(0,2).ixType().toIntVect();
    IntVect Bx_stag = warpx.getBfield_fp(0,0).ixType().toIntVect();
    IntVect By_stag = warpx.getBfield_fp(0,1).ixType().toIntVect();
    IntVect Bz_stag = warpx.getBfield_fp(0,2).ixType().toIntVect();
#ifdef WARPX_MAG_LLG
    IntVect Hx_stag = warpx.getHfield_fp(0,0).ixType().toIntVect();
    IntVect Hy_stag = warpx.getHfield_fp(0,1).ixType().toIntVect();
    IntVect Hz_stag = warpx.getHfield_fp(0,2).ixType().toIntVect();
    IntVect mag_alpha_stag = m_mag_alpha_mf->ixType().toIntVect();
    IntVect mag_gamma_stag = m_mag_gamma_mf->ixType().toIntVect();
    IntVect Mx_stag = warpx.getMfield_fp(0,0).ixType().toIntVect(); // face-centered
    IntVect My_stag = warpx.getMfield_fp(0,1).ixType().toIntVect();
    IntVect Mz_stag = warpx.getMfield_fp(0,2).ixType().toIntVect();
    IntVect mag_exchange_stag = m_mag_exchange_mf->ixType().toIntVect();
    IntVect mag_anisotropy_stag = m_mag_anisotropy_mf->ixType().toIntVect();
#endif


    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        Ex_IndexType[idim]      = Ex_stag[idim];
        Ey_IndexType[idim]      = Ey_stag[idim];
        Ez_IndexType[idim]      = Ez_stag[idim];
        Bx_IndexType[idim]      = Bx_stag[idim];
        By_IndexType[idim]      = By_stag[idim];
        Bz_IndexType[idim]      = Bz_stag[idim];
#ifdef WARPX_MAG_LLG
        Hx_IndexType[idim]             = Hx_stag[idim];
        Hy_IndexType[idim]             = Hy_stag[idim];
        Hz_IndexType[idim]             = Hz_stag[idim];
        mag_alpha_IndexType[idim]      = mag_alpha_stag[idim];
        mag_gamma_IndexType[idim]      = mag_gamma_stag[idim];
        Mx_IndexType[idim]             = Mx_stag[idim];
        My_IndexType[idim]             = My_stag[idim];
        Mz_IndexType[idim]             = Mz_stag[idim];
        mag_exchange_IndexType[idim]   = mag_exchange_stag[idim];
        mag_anisotropy_IndexType[idim] = mag_anisotropy_stag[idim];
        macro_cr_ratio[idim]           = 1;
#endif
    }
#if (AMREX_SPACEDIM==2)
        Ex_IndexType[2]      = 0;
        Ey_IndexType[2]      = 0;
        Ez_IndexType[2]      = 0;
        Bx_IndexType[2]      = 0;
        By_IndexType[2]      = 0;
        Bz_IndexType[2]      = 0;
#ifdef WARPX_MAG_LLG
        Hx_IndexType[2]              = 0;
        Hy_IndexType[2]              = 0;
        Hz_IndexType[2]              = 0;
        mag_alpha_IndexType[2]       = 0;
        mag_gamma_IndexType[2]       = 0;
        Mx_IndexType[2]              = 0;
        My_IndexType[2]              = 0;
        Mz_IndexType[2]              = 0;
        mag_exchange_IndexType[2]    = 0;
        mag_anisotropy_IndexType[2]  = 0;
        macro_cr_ratio[2]            = 1;
#endif
#endif


}

void
MacroscopicProperties::InitializeMacroMultiFabUsingParser (
                       MultiFab *macro_mf, ParserExecutor<3> const& macro_parser,
                       int lev)
{
#ifdef WARPX_MAG_LLG
    auto& warpx = WarpX::GetInstance();
    const auto dx_lev = warpx.Geom(lev).CellSizeArray();
    const RealBox& real_box = warpx.Geom(lev).ProbDomain();
    IntVect iv = macro_mf->ixType().toIntVect();
    for ( MFIter mfi(*macro_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        // Initialize ghost cells in addition to valid cells

        const Box& tb = mfi.tilebox(iv, macro_mf->nGrowVect());
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
#else
    amrex::ignore_unused(macro_mf, macro_parser, lev);
#endif
}
