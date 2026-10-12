#include "MacroscopicProperties.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/cnpy.h"
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>

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
#include <AMReX_Parser.H>

#include <AMReX_BaseFwd.H>

#include <memory>
#include <sstream>

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

    // Query mask index
    pp_macroscopic.query("npy_k_index", m_npy_k_index);
    pp_macroscopic.query("npy_k_index2", m_npy_k_index2);

    // Query input for material conductivity, sigma.
    pp_macroscopic.query("sigma_npy_value", m_sigma_npy_value);
    bool sigma_specified = false;
    bool sigma_func_specified = false;
    bool sigma_npy_specified = false;
    if (utils::parser::queryWithParser(pp_macroscopic, "sigma", m_sigma)) {
        m_sigma_s = "constant";
        sigma_specified = true;
    }
    if (pp_macroscopic.query("sigma_function(x,y,z)", m_str_sigma_function) ) {
        m_sigma_s = "parse_sigma_function";
        sigma_specified = true;
        sigma_func_specified = true;
    }
    if (pp_macroscopic.query("sigma_npy_file", m_sigma_npy_filename) ) {
        m_sigma_s = "parse_sigma_npy_file";
        sigma_specified = true;
        sigma_npy_specified = true;
        // parse second layer
        pp_macroscopic.query("sigma_npy_file2", m_sigma_npy_filename2);
    }
    if (sigma_func_specified && sigma_npy_specified) {
         // initialize both later in InitData
         m_sigma_s = "parse_sigma_both";
         sigma_specified = true;
    }
    if (!sigma_specified) {
        std::stringstream warnMsg;
        warnMsg << "Material conductivity is not specified. Using default vacuum value of "
                << m_sigma << " in the simulation.";
        ablastr::warn_manager::WMRecordWarning("Macroscopic properties", warnMsg.str());
    }
    // initialization of sigma (conductivity) with parser
    if (m_sigma_s == "parse_sigma_function" || m_sigma_s == "parse_sigma_both") {
        utils::parser::Store_parserString(
            pp_macroscopic, "sigma_function(x,y,z)", m_str_sigma_function);
        m_sigma_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_str_sigma_function,{"x","y","z"}));
    }

    // Query input for material permittivity, epsilon
    pp_macroscopic.query("epsilon_npy_value", m_epsilon_npy_value);
    bool epsilon_specified = false;
    bool epsilon_func_specified = false;
    bool epsilon_npy_specified = false;
    if (utils::parser::queryWithParser(pp_macroscopic, "epsilon", m_epsilon)) {
        m_epsilon_s = "constant";
        epsilon_specified = true;
    }
    if (pp_macroscopic.query("epsilon_function(x,y,z)", m_str_epsilon_function) ) {
        m_epsilon_s = "parse_epsilon_function";
        epsilon_specified = true;
        epsilon_func_specified = true;
    }
    if (pp_macroscopic.query("epsilon_npy_file", m_epsilon_npy_filename) ) {
        m_epsilon_s = "parse_epsilon_npy_file";
        epsilon_specified = true;
        epsilon_npy_specified = true;
        pp_macroscopic.query("epsilon_npy_file2", m_epsilon_npy_filename2);
    }
    if (epsilon_func_specified && epsilon_npy_specified) {
         // initialize both later in InitData
         m_epsilon_s = "parse_epsilon_both";
         epsilon_specified = true;
    }
    if (!epsilon_specified) {
        std::stringstream warnMsg;
        warnMsg << "Material conductivity is not specified. Using default vacuum value of "
                << m_epsilon << " in the simulation.";
        ablastr::warn_manager::WMRecordWarning("Macroscopic properties", warnMsg.str());
    }
    // initialization of epsilon (permittivity) with parser
    if (m_epsilon_s == "parse_epsilon_function" || m_epsilon_s == "parse_epsilon_both") {
        utils::parser::Store_parserString(
            pp_macroscopic, "epsilon_function(x,y,z)", m_str_epsilon_function);
        m_epsilon_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_str_epsilon_function,{"x","y","z"}));
    }

    // Query input for material permeability, mu
    pp_macroscopic.query("mu_npy_value", m_mu_npy_value);
    bool mu_specified = false;
    bool mu_func_specified = false;
    bool mu_npy_specified = false;
    if (utils::parser::queryWithParser(pp_macroscopic, "mu", m_mu)) {
        m_mu_s = "constant";
        mu_specified = true;
    }
    if (pp_macroscopic.query("mu_function(x,y,z)", m_str_mu_function) ) {
        m_mu_s = "parse_mu_function";
        mu_specified = true;
        mu_func_specified = true;
    }
    if (pp_macroscopic.query("mu_npy_file", m_mu_npy_filename) ) {
        m_mu_s = "parse_mu_npy_file";
        mu_specified = true;
        mu_npy_specified = true;
        pp_macroscopic.query("mu_npy_file2", m_mu_npy_filename2);
    }
    if (mu_func_specified && mu_npy_specified) {
         // initialize both later in InitData
         m_mu_s = "parse_mu_both";
         mu_specified = true;
    }
    if (!mu_specified) {
        std::stringstream warnMsg;
        warnMsg << "Material conductivity is not specified. Using default vacuum value of "
                << m_mu << " in the simulation.";
        ablastr::warn_manager::WMRecordWarning("Macroscopic properties", warnMsg.str());
    }
    // initialization of mu (permeability) with parser
    if (m_mu_s == "parse_mu_function" || m_mu_s == "parse_mu_both") {
        utils::parser::Store_parserString(
            pp_macroscopic, "mu_function(x,y,z)", m_str_mu_function);
        m_mu_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_str_mu_function,{"x","y","z"}));
    }

#ifdef WARPX_MAG_LLG
    auto &warpx = WarpX::GetInstance();
    pp_macroscopic.get("mag_Ms_init_style", m_mag_Ms_s);
    if (m_mag_Ms_s == "constant") pp_macroscopic.get("mag_Ms", m_mag_Ms);
    // _mag_ such that it's clear the Ms variable is only meaningful for magnetic materials
    //initialization with parser
    if (m_mag_Ms_s == "parse_mag_Ms_function") {
        utils::parser::Store_parserString(pp_macroscopic, "mag_Ms_function(x,y,z)", m_str_mag_Ms_function);
        m_mag_Ms_parser = std::make_unique<amrex::Parser>(
                                  utils::parser::makeParser(m_str_mag_Ms_function,{"x","y","z"}));
    }

    pp_macroscopic.get("mag_alpha_init_style", m_mag_alpha_s);
    if (m_mag_alpha_s == "constant") pp_macroscopic.get("mag_alpha", m_mag_alpha);
    // _mag_ such that it's clear the alpha variable is only meaningful for magnetic materials
    //initialization with parser
    if (m_mag_alpha_s == "parse_mag_alpha_function") {
        utils::parser::Store_parserString(pp_macroscopic, "mag_alpha_function(x,y,z)", m_str_mag_alpha_function);
        m_mag_alpha_parser = std::make_unique<amrex::Parser>(
                                  utils::parser::makeParser(m_str_mag_alpha_function,{"x","y","z"}));
    }

    pp_macroscopic.get("mag_gamma_init_style", m_mag_gamma_s);
    if (m_mag_gamma_s == "constant") pp_macroscopic.get("mag_gamma", m_mag_gamma);
    // _mag_ such that it's clear the gamma variable parsed here is only meaningful for magnetic materials
    //initialization with parser
    if (m_mag_gamma_s == "parse_mag_gamma_function") {
        utils::parser::Store_parserString(pp_macroscopic, "mag_gamma_function(x,y,z)", m_str_mag_gamma_function);
        m_mag_gamma_parser = std::make_unique<amrex::Parser>(
                                  utils::parser::makeParser(m_str_mag_gamma_function,{"x","y","z"}));
    }

    if (warpx.mag_LLG_exchange_coupling == 1) { // spin exchange coupling turned off by default
        pp_macroscopic.get("mag_exchange_init_style", m_mag_exchange_s);
        if (m_mag_exchange_s == "constant") pp_macroscopic.get("mag_exchange", m_mag_exchange);
        // _mag_ such that it's clear the exch variable is only meaningful for magnetic materials
        //initialization with parser
        if (m_mag_exchange_s == "parse_mag_exchange_function") {
            utils::parser::Store_parserString(pp_macroscopic, "mag_exchange_function(x,y,z)", m_str_mag_exchange_function);
            m_mag_exchange_parser = std::make_unique<amrex::Parser>(
                                      utils::parser::makeParser(m_str_mag_exchange_function,{"x","y","z"}));
        }
    }

    if (warpx.mag_LLG_anisotropy_coupling == 1) { // magnetic crystal is considered as isotropic by default
        pp_macroscopic.get("mag_anisotropy_init_style", m_mag_anisotropy_s);
        if (m_mag_anisotropy_s == "constant") pp_macroscopic.get("mag_anisotropy", m_mag_anisotropy);
        // _mag_ such that it's clear the exch variable is only meaningful for magnetic materials
        //initialization with parser
        if (m_mag_anisotropy_s == "parse_mag_anisotropy_function") {
            utils::parser::Store_parserString(pp_macroscopic, "mag_anisotropy_function(x,y,z)", m_str_mag_anisotropy_function);
            m_mag_anisotropy_parser = std::make_unique<amrex::Parser>(
                                      utils::parser::makeParser(m_str_mag_anisotropy_function,{"x","y","z"}));
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
    amrex::Print() << Utils::TextMsg::Info("we are in init data of macro");
    auto & warpx = WarpX::GetInstance();
    // Get BoxArray and DistributionMap of warpx instance.
    int lev = 0;
    amrex::BoxArray ba = warpx.boxArray(lev);
    amrex::DistributionMapping dmap = warpx.DistributionMap(lev);
    const amrex::IntVect ng_EB_alloc = warpx.getngEB();
    // Define material property multifabs using ba and dmap from WarpX instance
    // sigma is cell-centered MultiFab
    m_sigma_mf = std::make_unique<amrex::MultiFab>(ba, dmap, 1, ng_EB_alloc);
    // epsilon is cell-centered MultiFab
    m_eps_mf = std::make_unique<amrex::MultiFab>(ba, dmap, 1, ng_EB_alloc);
    // mu is cell-centered MultiFab
    m_mu_mf = std::make_unique<amrex::MultiFab>(ba, dmap, 1, ng_EB_alloc);

    ////////////////////////
    // PEC mask

    // call this before setting sigma in the rest of the domain since this zeros sigma once the PEC mask is defined
    if (warpx.use_PEC_mask) {

        // read in the layer(s) from the numpy file(s)
        if (m_sigma_s == "parse_sigma_npy_file" || m_sigma_s == "parse_sigma_both") {
            InitializeMacroMultiFabFromNumpy(m_sigma_mf.get(), m_sigma_npy_filename, lev, m_npy_k_index, m_sigma_npy_value);
            if (!m_sigma_npy_filename2.empty()) {
                InitializeMacroMultiFabFromNumpy(m_sigma_mf.get(), m_sigma_npy_filename2, lev, m_npy_k_index2, m_sigma_npy_value);
            }
        } else {
            amrex::Abort("warpx.use_PEC_mask requies a npy file");
        }
        // get pointer to PEC mask in WarpX class
        amrex::MultiFab * PECx = warpx.get_pointer_PEC_fp(lev,0);
        amrex::MultiFab * PECy = warpx.get_pointer_PEC_fp(lev,1);
        amrex::MultiFab * PECz = warpx.get_pointer_PEC_fp(lev,2);

        PECx->setVal(1.);
        PECy->setVal(1.);
        PECz->setVal(1.);

        InitializePECFromSigma(m_sigma_mf.get(), PECx, PECy, m_npy_k_index);
        if (!m_sigma_npy_filename2.empty()) {
            InitializePECFromSigma(m_sigma_mf.get(), PECx, PECy, m_npy_k_index2);
        }
        // zero sigma
        m_sigma_mf->setVal(0.);
    }

    ////////////////////////
    // Initialize sigma
    if (m_sigma_s == "constant") {

        m_sigma_mf->setVal(m_sigma);

    }
    // Step 1: Initialize sigma from parse (valid and ghost region)
    if (m_sigma_s == "parse_sigma_function" || m_sigma_s == "parse_sigma_both") {
        InitializeMacroMultiFabUsingParser(m_sigma_mf.get(), m_sigma_parser->compile<3>(), lev);
    }

    // Step 2: If not using PEC mask, overwrite with numpy mask in valid region if provided
    if (!warpx.use_PEC_mask) {
        if (m_sigma_s == "parse_sigma_npy_file" || m_sigma_s == "parse_sigma_both") {
            InitializeMacroMultiFabFromNumpy(m_sigma_mf.get(), m_sigma_npy_filename, lev, m_npy_k_index, m_sigma_npy_value);
            if (!m_sigma_npy_filename2.empty()) {
                InitializeMacroMultiFabFromNumpy(m_sigma_mf.get(), m_sigma_npy_filename2, lev, m_npy_k_index2, m_sigma_npy_value);
            }
        }
    }

    ////////////////////////
    // Initialize epsilon
    if (m_epsilon_s == "constant") {

        m_eps_mf->setVal(m_epsilon);

    }
    // Step 1: Initialize epsilon from parse (valid and ghost region)
    if (m_epsilon_s == "parse_epsilon_function" || m_epsilon_s == "parse_epsilon_both") {
        InitializeMacroMultiFabUsingParser(m_eps_mf.get(), m_epsilon_parser->compile<3>(), lev);
    }

    // Step 2: Overwrite with numpy mask in valid region if provided
    if (m_epsilon_s == "parse_epsilon_npy_file" || m_epsilon_s == "parse_epsilon_both") {
        InitializeMacroMultiFabFromNumpy(m_eps_mf.get(), m_epsilon_npy_filename, lev, m_npy_k_index, m_epsilon_npy_value);
        if (!m_epsilon_npy_filename2.empty()) {
            InitializeMacroMultiFabFromNumpy(m_eps_mf.get(), m_epsilon_npy_filename2, lev, m_npy_k_index2, m_epsilon_npy_value);
        }
    }

    ////////////////////////
    // Initialize mu
    if (m_mu_s == "constant") {

        m_mu_mf->setVal(m_mu);

    }
    // Step 1: Initialize mu from parse (valid and ghost region)
    if (m_mu_s == "parse_mu_function" || m_mu_s == "parse_mu_both") {
        InitializeMacroMultiFabUsingParser(m_mu_mf.get(), m_mu_parser->compile<3>(), lev);
    }

    // Step 2: Overwrite with numpy mask in valid region if provided
    if (m_mu_s == "parse_mu_npy_file" || m_mu_s == "parse_mu_both") {
        InitializeMacroMultiFabFromNumpy(m_mu_mf.get(), m_mu_npy_filename, lev, m_npy_k_index, m_mu_npy_value);
        if (!m_mu_npy_filename2.empty()) {
            InitializeMacroMultiFabFromNumpy(m_mu_mf.get(), m_mu_npy_filename2, lev, m_npy_k_index2, m_mu_npy_value);
        }
    }

#ifdef WARPX_MAG_LLG

    // all magnetic macroparameters are stored on faces
    for (int i=0; i<3; ++i) {
        m_mag_Ms_mf[i]         = std::make_unique<MultiFab>(amrex::convert(ba,IntVect::TheDimensionVector(i)), dmap, 1, ng_EB_alloc);
        m_mag_alpha_mf[i]      = std::make_unique<MultiFab>(amrex::convert(ba,IntVect::TheDimensionVector(i)), dmap, 1, ng_EB_alloc);
        m_mag_gamma_mf[i]      = std::make_unique<MultiFab>(amrex::convert(ba,IntVect::TheDimensionVector(i)), dmap, 1, ng_EB_alloc);
        m_mag_exchange_mf[i]   = std::make_unique<MultiFab>(amrex::convert(ba,IntVect::TheDimensionVector(i)), dmap, 1, ng_EB_alloc);
        m_mag_anisotropy_mf[i] = std::make_unique<MultiFab>(amrex::convert(ba,IntVect::TheDimensionVector(i)), dmap, 1, ng_EB_alloc);
    }

    // mag_Ms - defined at cell centers
    if (m_mag_Ms_s == "constant") {
        m_mag_Ms_mf[0]->setVal(m_mag_Ms);
        m_mag_Ms_mf[1]->setVal(m_mag_Ms);
        m_mag_Ms_mf[2]->setVal(m_mag_Ms);
    }
    else if (m_mag_Ms_s == "parse_mag_Ms_function"){
        InitializeMacroMultiFabUsingParser(m_mag_Ms_mf[0].get(), m_mag_Ms_parser->compile<3>(), lev);
        InitializeMacroMultiFabUsingParser(m_mag_Ms_mf[1].get(), m_mag_Ms_parser->compile<3>(), lev);
        InitializeMacroMultiFabUsingParser(m_mag_Ms_mf[2].get(), m_mag_Ms_parser->compile<3>(), lev);
    }
    // if there are regions with Ms=0, the user must provide mur value there
    for (int i=0; i<3; ++i) {
        if (m_mag_Ms_mf[i]->min(0,m_mag_Ms_mf[i]->nGrow()) < 0._rt){
            amrex::Abort("Ms must be non-negative values");
        }
    }
    for (int i=0; i<3; ++i) {
        if (m_mag_Ms_mf[i]->min(0,m_mag_Ms_mf[i]->nGrow()) == 0._rt){
            if (m_mu_s != "constant" && m_mu_s != "parse_mu_function"){
                amrex::Abort("permeability must be specified since part of the simulation domain is non-magnetic !");
            }
        }
    }

    // mag_alpha - defined at faces
    if (m_mag_alpha_s == "constant") {
        m_mag_alpha_mf[0]->setVal(m_mag_alpha);
        m_mag_alpha_mf[1]->setVal(m_mag_alpha);
        m_mag_alpha_mf[2]->setVal(m_mag_alpha);
    }
    else if (m_mag_alpha_s == "parse_mag_alpha_function"){
        InitializeMacroMultiFabUsingParser(m_mag_alpha_mf[0].get(), m_mag_alpha_parser->compile<3>(), lev);
        InitializeMacroMultiFabUsingParser(m_mag_alpha_mf[1].get(), m_mag_alpha_parser->compile<3>(), lev);
        InitializeMacroMultiFabUsingParser(m_mag_alpha_mf[2].get(), m_mag_alpha_parser->compile<3>(), lev);
    }
    for (int i=0; i<3; ++i) {
        if (m_mag_alpha_mf[i]->min(0,m_mag_alpha_mf[i]->nGrow()) < 0._rt) {
            amrex::Abort("alpha should be positive, but the user input has negative values");
        }
    }

    // mag_gamma - defined at faces
    if (m_mag_gamma_s == "constant") {
        m_mag_gamma_mf[0]->setVal(m_mag_gamma);
        m_mag_gamma_mf[1]->setVal(m_mag_gamma);
        m_mag_gamma_mf[2]->setVal(m_mag_gamma);
    }
    else if (m_mag_gamma_s == "parse_mag_gamma_function"){
        InitializeMacroMultiFabUsingParser(m_mag_gamma_mf[0].get(), m_mag_gamma_parser->compile<3>(), lev);
        InitializeMacroMultiFabUsingParser(m_mag_gamma_mf[1].get(), m_mag_gamma_parser->compile<3>(), lev);
        InitializeMacroMultiFabUsingParser(m_mag_gamma_mf[2].get(), m_mag_gamma_parser->compile<3>(), lev);
    }
    for (int i=0; i<3; ++i) {
        if (m_mag_gamma_mf[i]->min(0,m_mag_gamma_mf[i]->nGrow()) > 0._rt) {
            amrex::Abort("gamma should be negative, but the user input has positive values");
        }
    }

    // mag_exchange - defined at faces
    if (m_mag_exchange_s == "constant") {
        m_mag_exchange_mf[0]->setVal(m_mag_exchange);
        m_mag_exchange_mf[1]->setVal(m_mag_exchange);
        m_mag_exchange_mf[2]->setVal(m_mag_exchange);
    }
    else if (m_mag_exchange_s == "parse_mag_exchange_function"){
        InitializeMacroMultiFabUsingParser(m_mag_exchange_mf[0].get(), m_mag_exchange_parser->compile<3>(), lev);
        InitializeMacroMultiFabUsingParser(m_mag_exchange_mf[1].get(), m_mag_exchange_parser->compile<3>(), lev);
        InitializeMacroMultiFabUsingParser(m_mag_exchange_mf[2].get(), m_mag_exchange_parser->compile<3>(), lev);
    }

    // mag_anisotropy - defined at faces
    if (m_mag_anisotropy_s == "constant") {
        m_mag_anisotropy_mf[0]->setVal(m_mag_anisotropy);
        m_mag_anisotropy_mf[1]->setVal(m_mag_anisotropy);
        m_mag_anisotropy_mf[2]->setVal(m_mag_anisotropy);
    }
    else if (m_mag_anisotropy_s == "parse_mag_anisotropy_function"){
        InitializeMacroMultiFabUsingParser(m_mag_anisotropy_mf[0].get(), m_mag_anisotropy_parser->compile<3>(), lev);
        InitializeMacroMultiFabUsingParser(m_mag_anisotropy_mf[1].get(), m_mag_anisotropy_parser->compile<3>(), lev);
        InitializeMacroMultiFabUsingParser(m_mag_anisotropy_mf[2].get(), m_mag_anisotropy_parser->compile<3>(), lev);
    }
#endif


    amrex::IntVect sigma_stag = m_sigma_mf->ixType().toIntVect();
    amrex::IntVect epsilon_stag = m_eps_mf->ixType().toIntVect();
    amrex::IntVect mu_stag = m_mu_mf->ixType().toIntVect();
    amrex::IntVect Ex_stag = warpx.getEfield_fp(0,0).ixType().toIntVect();
    amrex::IntVect Ey_stag = warpx.getEfield_fp(0,1).ixType().toIntVect();
    amrex::IntVect Ez_stag = warpx.getEfield_fp(0,2).ixType().toIntVect();
    IntVect Bx_stag = warpx.getBfield_fp(0,0).ixType().toIntVect();
    IntVect By_stag = warpx.getBfield_fp(0,1).ixType().toIntVect();
    IntVect Bz_stag = warpx.getBfield_fp(0,2).ixType().toIntVect();
#ifdef WARPX_MAG_LLG
    IntVect Hx_stag = warpx.getHfield_fp(0,0).ixType().toIntVect();
    IntVect Hy_stag = warpx.getHfield_fp(0,1).ixType().toIntVect();
    IntVect Hz_stag = warpx.getHfield_fp(0,2).ixType().toIntVect();
    IntVect Mx_stag = warpx.getMfield_fp(0,0).ixType().toIntVect();
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
        Bx_IndexType[idim]      = Bx_stag[idim];
        By_IndexType[idim]      = By_stag[idim];
        Bz_IndexType[idim]      = Bz_stag[idim];
        macro_cr_ratio[idim]    = 1;
#ifdef WARPX_MAG_LLG
        Hx_IndexType[idim] = Hx_stag[idim];
        Hy_IndexType[idim] = Hy_stag[idim];
        Hz_IndexType[idim] = Hz_stag[idim];
        Mx_IndexType[idim] = Mx_stag[idim];
        My_IndexType[idim] = My_stag[idim];
        Mz_IndexType[idim] = Mz_stag[idim];
#endif
    }
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        sigma_IndexType[2]   = 0;
        epsilon_IndexType[2] = 0;
        mu_IndexType[2]      = 0;
        Ex_IndexType[2]      = 0;
        Ey_IndexType[2]      = 0;
        Ez_IndexType[2]      = 0;
        Bx_IndexType[2]      = 0;
        By_IndexType[2]      = 0;
        Bz_IndexType[2]      = 0;
        macro_cr_ratio[2]    = 0;
#ifdef WARPX_MAG_LLG
        Hx_IndexType[2]              = 0;
        Hy_IndexType[2]              = 0;
        Hz_IndexType[2]              = 0;
        Mx_IndexType[2]              = 0;
        My_IndexType[2]              = 0;
        Mz_IndexType[2]              = 0;
#endif
#endif
}

void
MacroscopicProperties::InitializeMacroMultiFabUsingParser (
                       amrex::MultiFab *macro_mf,
                       amrex::ParserExecutor<3> const& macro_parser,
                       const int lev)
{
    WarpX& warpx = WarpX::GetInstance();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev = warpx.Geom(lev).CellSizeArray();
    const amrex::RealBox& real_box = warpx.Geom(lev).ProbDomain();
    amrex::IntVect iv = macro_mf->ixType().toIntVect();
    for ( amrex::MFIter mfi(*macro_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        // Initialize ghost cells in addition to valid cells

        const amrex::Box& tb = mfi.tilebox( iv, macro_mf->nGrowVect());
        amrex::Array4<amrex::Real> const& macro_fab =  macro_mf->array(mfi);
        amrex::ParallelFor (tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Shift x, y, z position based on index type

#if defined(WARPX_DIM_1D_Z)
                const amrex::Real x = 0._rt;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real z = j * dx_lev[0] + real_box.lo(0) + fac_z;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                const amrex::Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real z = j * dx_lev[1] + real_box.lo(1) + fac_z;
#else
                const amrex::Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;
                const amrex::Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
                const amrex::Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                // initialize the macroparameter
                macro_fab(i,j,k) = macro_parser(x,y,z);
        });

    }
}

void
MacroscopicProperties::InitializeMacroMultiFabFromNumpy (
    amrex::MultiFab* macro_mf,
    const std::string& npy_filename,
    const int lev,
    const int npy_k_index,
    const amrex::Real npy_value)
{

    // Load numpy array
    cnpy::NpyArray arr = cnpy::npy_load(npy_filename);
    amrex::Print() << "Loaded numpy array: " << npy_filename << "\n";

    // Check expected dimensions (assume 3D)
    if (arr.shape.size() != 3) {
        amrex::Abort("Expected a 3D numpy array for initialization.");
    }

    const size_t nx = arr.shape[0];
    const size_t ny = arr.shape[1];
    const size_t nz = arr.shape[2];

    // Validate array size against domain
    WarpX& warpx = WarpX::GetInstance();
    const amrex::Box domain_box = warpx.Geom(lev).Domain();
    const amrex::IntVect dom_size = domain_box.size();

    // if (dom_size[0] != static_cast<int>(nx) ||
    //     dom_size[1] != static_cast<int>(ny) ||
    //     dom_size[2] != static_cast<int>(nz)) {
    //     amrex::Abort("Mismatch between numpy array shape and simulation domain size!");
    // }

    if (dom_size[0] != static_cast<int>(nx) ||
        dom_size[1] != static_cast<int>(ny)) {
        amrex::Abort("Mismatch between numpy array shape and simulation domain size!");
    }


    // const double* data = arr.data<double>();
    const double* data = arr.data<double>();
    std::size_t total_size = nx * ny * nz;

    // Convert to amrex::Real on host
    std::vector<amrex::Real> host_data(total_size);
    for (std::size_t i = 0; i < total_size; ++i) {
        host_data[i] = static_cast<amrex::Real>(data[i]);
    }

    // Copy to device
    amrex::Gpu::DeviceVector<amrex::Real> data_dev;
    data_dev.resize(total_size);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_data.begin(), host_data.end(), data_dev.begin());
    const amrex::Real* dptr = data_dev.data();

    amrex::IntVect iv = macro_mf->ixType().toIntVect();

    for (MFIter mfi(*macro_mf); mfi.isValid(); ++mfi) {
        // we need to loop over ghost cells since the PML algorithm creates temporary grids with valid
        // and ghost regions; some may lay on the interior domain and some may lie outside the domain
        const Box& bx = mfi.tilebox( iv, macro_mf->nGrowVect());

        /*
        if (mfi.index() == 0) {
            const auto lo = bx.smallEnd();
            const auto hi = bx.bigEnd();
            const auto size = bx.size();

            amrex::Print() << "fab shape: ["
                           << size[0] << ", "
                           << size[1] << ", "
                           << size[2] << "], index range: ["
                           << lo[0] << ":" << hi[0] << ", "
                           << lo[1] << ":" << hi[1] << ", "
                           << lo[2] << ":" << hi[2] << "]\n";
        }
        */

        Array4<amrex::Real> fab = macro_mf->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

            int idx2d;
            // only overwrite macro value from the gds file if you are in the valid region
            if (k == npy_k_index) {

                if (i < 0 && j >= 0 && j < dom_size[1]) {
                    // left edge PML
                    idx2d = 0 * ny + j;
                } else if (i >= dom_size[0] && j >= 0 && j < dom_size[1]) {
                    // right edge PML
                    idx2d = (dom_size[0]-1) * ny + j;
                } else if (i >= 0 && i < dom_size[0] && j < 0) {
                    // bottom edge PML
                    idx2d = i * ny + 0;
                } else if (i >= 0 && i < dom_size[0] && j >= dom_size[1]) {
                    // top edge PML
                    idx2d = i * ny + (dom_size[1]-1);
                } else if (i < 0 && j < 0) {
                    // bottom left corner PML
                    idx2d = 0 * ny + 0;
                } else if (i < 0 && j >= dom_size[1]) {
                    // top left corner PML
                    idx2d = 0 * ny + (dom_size[1]-1);
                } else if (i >= dom_size[0] && j < 0) {
                    // bottom right corner PML
                    idx2d = (dom_size[0]-1) * ny + 0;
                } else if (i >= dom_size[0] && j >= dom_size[1]) {
                    // top right corner PML
                    idx2d = (dom_size[0]-1) * ny + (dom_size[1]-1);
                } else {
                    // valid region PML
                    idx2d = i * ny + j;
                }

                // if the mask value is 1, set macroscopic value to npy_value
                if (dptr[idx2d] == 1) {
                    fab(i, j, k) = npy_value;
                }
            }

        });
    }
    amrex::Gpu::streamSynchronize();
}

void
MacroscopicProperties::InitializePECFromSigma (amrex::MultiFab* sigma_mf,
                                               amrex::MultiFab* PECx,
                                               amrex::MultiFab* PECy,
                                               const int npy_k_index)
{

    int nComp = PECx->nComp();

    // PEC for Ex is on yz edges
    for (MFIter mfi(*PECx); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        Array4<amrex::Real> sigma = sigma_mf->array(mfi);
        Array4<amrex::Real> Px = PECx->array(mfi);

        ParallelFor(bx, nComp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {

            if (k == npy_k_index) {
                if (sigma(i,j,k) != 0. || sigma(i,j-1,k) != 0) {
                    Px(i,j,k,n) = 0.;
                }
            }

        });
    }

    // PEC for Ey is on xz edges
    for (MFIter mfi(*PECy); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        Array4<amrex::Real> sigma = sigma_mf->array(mfi);
        Array4<amrex::Real> Py = PECy->array(mfi);

        ParallelFor(bx, nComp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {

            if (k == npy_k_index) {
                if (sigma(i,j,k) != 0. || sigma(i-1,j,k) != 0) {
                    Py(i,j,k,n) = 0.;
                }
            }

        });
    }
}
