#include "MacroscopicProperties.H"
#include <AMReX_ParmParse.H>
#include "WarpX.H"

using namespace amrex;

MacroscopicProperties::MacroscopicProperties ()
{
    ReadParameters();
}

void
MacroscopicProperties::ReadParameters ()
{
    ParmParse pp("macroscopic");
    // Since macroscopic maxwell solve is turned on, user must define sigma, mu, and epsilon //

    pp.get("sigma_init_style", m_sigma_s);
    // constant initialization
    if (m_sigma_s == "constant") pp.get("sigma", m_sigma);
    // initialization with parser
    if (m_sigma_s == "parse_sigma_function") {
        Store_parserString(pp, "sigma_function(x,y,z)", m_str_sigma_function);
        m_sigma_parser.reset(new ParserWrapper<3>(
                                 makeParser(m_str_sigma_function,{"x","y","z"}) ) );
    }

    pp.get("epsilon_init_style", m_epsilon_s);
    if (m_epsilon_s == "constant") pp.get("epsilon", m_epsilon);
    // initialization with parser
    if (m_epsilon_s == "parse_epsilon_function") {
        Store_parserString(pp, "epsilon_function(x,y,z)", m_str_epsilon_function);
        m_epsilon_parser.reset(new ParserWrapper<3>(
                                 makeParser(m_str_epsilon_function,{"x","y","z"}) ) );
    }

    pp.get("mu_init_style", m_mu_s);
    if (m_mu_s == "constant") pp.get("mu", m_mu);
    // initialization with parser
    if (m_mu_s == "parse_mu_function") {
        Store_parserString(pp, "mu_function(x,y,z)", m_str_mu_function);
        m_mu_parser.reset(new ParserWrapper<3>(
                                 makeParser(m_str_mu_function,{"x","y","z"}) ) );
    }

#ifdef WARPX_MAG_LLG
    pp.get("mag_Ms_init_style", m_mag_Ms_s);
    if (m_mag_Ms_s == "constant") pp.get("mag_Ms", m_mag_Ms);
    // _mag_ such that it's clear the Ms variable is only meaningful for magnetic materials
    //initialization with parser
    if (m_mag_Ms_s == "parse_mag_Ms_function") {
        Store_parserString(pp, "mag_Ms_function(x,y,z)", m_str_mag_Ms_function);
        m_mag_Ms_parser.reset(new ParserWrapper<3>(
                                  makeParser(m_str_mag_Ms_function,{"x","y","z"})));
    }

    pp.get("mag_alpha_init_style", m_mag_alpha_s);
    if (m_mag_alpha_s == "constant") pp.get("mag_alpha", m_mag_alpha);
    // _mag_ such that it's clear the alpha variable is only meaningful for magnetic materials
    //initialization with parser
    if (m_mag_alpha_s == "parse_mag_alpha_function") {
        Store_parserString(pp, "mag_alpha_function(x,y,z)", m_str_mag_alpha_function);
        m_mag_alpha_parser.reset(new ParserWrapper<3>(
                                  makeParser(m_str_mag_alpha_function,{"x","y","z"})));
    }

    pp.get("mag_gamma_init_style", m_mag_gamma_s);
    if (m_mag_gamma_s == "constant") pp.get("mag_gamma", m_mag_gamma);
    // _mag_ such that it's clear the gamma variable parsed here is only meaningful for magnetic materials
    //initialization with parser
    if (m_mag_gamma_s == "parse_mag_gamma_function") {
        Store_parserString(pp, "mag_gamma_function(x,y,z)", m_str_mag_gamma_function);
        m_mag_gamma_parser.reset(new ParserWrapper<3>(
                                  makeParser(m_str_mag_gamma_function,{"x","y","z"})));
    }

    m_mag_normalized_error = 0.1;
    pp.query("mag_normalized_error",m_mag_normalized_error);

    m_mag_max_iter = 100;
    pp.query("mag_max_iter",m_mag_max_iter);

    m_mag_tol = 0.0001;
    pp.query("mag_tol",m_mag_tol);

#endif
}

void
MacroscopicProperties::InitData ()
{
    amrex::Print() << "we are in init data of macro \n";
    auto & warpx = WarpX::GetInstance();

    // Get BoxArray and DistributionMap of warpX.
    int lev = 0;
    BoxArray ba = warpx.boxArray(lev);
    DistributionMapping dmap = warpx.DistributionMap(lev);
    int ng = 3;
      // allocate multifabs using ba and dmap from WarpX instance
    m_sigma_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng); // cell-centered
    m_eps_mf = std::make_unique<MultiFab>(ba, dmap, 1, ng);
    m_mu_mf = std::make_unique<MultiFab>(amrex::convert(ba,amrex::IntVect::TheUnitVector()), dmap, 1, ng);

#ifdef WARPX_MAG_LLG
    // all magnetic macroparameters are stored on cell nodes
    m_mag_Ms_mf = std::make_unique<MultiFab>(amrex::convert(ba,amrex::IntVect::TheUnitVector()), dmap, 1, ng);
    m_mag_alpha_mf = std::make_unique<MultiFab>(amrex::convert(ba,amrex::IntVect::TheUnitVector()), dmap, 1, ng);
    m_mag_gamma_mf = std::make_unique<MultiFab>(amrex::convert(ba,amrex::IntVect::TheUnitVector()), dmap, 1, ng);
#endif

    if (m_sigma_s == "constant") {

        m_sigma_mf->setVal(m_sigma);

    } else if (m_sigma_s == "parse_sigma_function") {

        InitializeMacroMultiFabUsingParser(m_sigma_mf.get(), m_sigma_parser.get(), lev);
    }
      // eps - cell-centered
    if (m_epsilon_s == "constant") {

        m_eps_mf->setVal(m_epsilon);

    } else if (m_epsilon_s == "parse_epsilon_function") {

        InitializeMacroMultiFabUsingParser(m_eps_mf.get(), m_epsilon_parser.get(), lev);

    }
      // mu - node-based
    if (m_mu_s == "constant") {

        m_mu_mf->setVal(m_mu);

    } else if (m_mu_s == "parse_mu_function") {

        InitializeMacroMultiFabUsingParser(m_mu_mf.get(), m_mu_parser.get(), lev);

    }

#ifdef WARPX_MAG_LLG
    // mag_Ms - defined at node
    if (m_mag_Ms_s == "constant") {
        m_mag_Ms_mf->setVal(m_mag_Ms);
    }
    else if (m_mag_Ms_s == "parse_mag_Ms_function"){
        InitializeMacroMultiFabUsingParser(m_mag_Ms_mf.get(), m_mag_Ms_parser.get(), lev);
    }

    // mag_alpha - defined at node
    if (m_mag_alpha_s == "constant") {
        m_mag_alpha_mf->setVal(m_mag_alpha);
    }
    else if (m_mag_alpha_s == "parse_mag_alpha_function"){
        InitializeMacroMultiFabUsingParser(m_mag_alpha_mf.get(), m_mag_alpha_parser.get(), lev);
    }
    if (m_mag_alpha_mf->min(0,m_mag_alpha_mf->nGrow()) < 0) {
        amrex::Abort("alpha should be positive, but the user input has negative values");
    }

    // mag_gamma - defined at node
    if (m_mag_gamma_s == "constant") {
        m_mag_gamma_mf->setVal(m_mag_gamma);

    }
    else if (m_mag_gamma_s == "parse_mag_gamma_function"){
        InitializeMacroMultiFabUsingParser(m_mag_gamma_mf.get(), m_mag_gamma_parser.get(), lev);
    }
    if (m_mag_gamma_mf->max(0,m_mag_gamma_mf->nGrow()) > 0) {
        amrex::Abort("gamma should be negative, but the user input has positive values");
    }
#endif
}

void
MacroscopicProperties::InitializeMacroMultiFabUsingParser (
                       MultiFab *macro_mf, ParserWrapper<3> *macro_parser,
                       int lev)
{
    auto& warpx = WarpX::GetInstance();
    const auto dx_lev = warpx.Geom(lev).CellSizeArray();
    const RealBox& real_box = warpx.Geom(lev).ProbDomain();
    IntVect iv = macro_mf->ixType().toIntVect();
    for ( MFIter mfi(*macro_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Initialize ghost cells in addition to valid cells by calling nGrow()
        const Box& tb = mfi.tilebox(iv, macro_mf->nGrowVect() );

        auto const& macro_fab =  macro_mf->array(mfi);

        amrex::ParallelFor (tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Shift x, y, z position based on index type
                Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;

                Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;

                Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
                Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;

                // initialize the macroparameter
                macro_fab(i,j,k) = (*macro_parser)(x,y,z);
        });

    }


}
