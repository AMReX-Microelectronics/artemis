/* Copyright 2019-2020 Andrew Myers, Ann Almgren, Aurore Blelly
 * Axel Huebl, Burlen Loring, Maxence Thevenet
 * Michael Rowan, Remi Lehe, Revathi Jambunathan
 * Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"
#include "Filter/BilinearFilter.H"
#include "Filter/NCIGodfreyFilter.H"
#include "Parser/GpuParser.H"
#include "Utils/WarpXUtil.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#ifdef BL_USE_SENSEI_INSITU
#   include <AMReX_AmrMeshInSituBridge.H>
#endif

#include <memory>

using namespace amrex;

void
WarpX::PostProcessBaseGrids (BoxArray& ba0) const
{
    if (numprocs != 0) {
        const Box& dom = Geom(0).Domain();
        const IntVect& domlo = dom.smallEnd();
        const IntVect& domlen = dom.size();
        const IntVect sz = domlen / numprocs;
        const IntVect extra = domlen - sz*numprocs;
        BoxList bl;
#if (AMREX_SPACEDIM == 3)
        for (int k = 0; k < numprocs[2]; ++k) {
            // The first extra[2] blocks get one extra cell with a total of
            // sz[2]+1.  The rest get sz[2] cells.  The docomposition in y
            // and x directions are similar.
            int klo = (k < extra[2]) ? k*(sz[2]+1) : (k*sz[2]+extra[2]);
            int khi = (k < extra[2]) ? klo+(sz[2]+1)-1 : klo+sz[2]-1;
            klo += domlo[2];
            khi += domlo[2];
#endif
            for (int j = 0; j < numprocs[1]; ++j) {
                int jlo = (j < extra[1]) ? j*(sz[1]+1) : (j*sz[1]+extra[1]);
                int jhi = (j < extra[1]) ? jlo+(sz[1]+1)-1 : jlo+sz[1]-1;
                jlo += domlo[1];
                jhi += domlo[1];
                for (int i = 0; i < numprocs[0]; ++i) {
                    int ilo = (i < extra[0]) ? i*(sz[0]+1) : (i*sz[0]+extra[0]);
                    int ihi = (i < extra[0]) ? ilo+(sz[0]+1)-1 : ilo+sz[0]-1;
                    ilo += domlo[0];
                    ihi += domlo[0];
                    bl.push_back(Box(IntVect(AMREX_D_DECL(ilo,jlo,klo)),
                                     IntVect(AMREX_D_DECL(ihi,jhi,khi))));
        AMREX_D_TERM(},},})
        ba0 = BoxArray(std::move(bl));
    }
}

void
WarpX::InitData ()
{
    WARPX_PROFILE("WarpX::InitData()");
    Print() << "WarpX (" << WarpX::Version() << ")\n";
#ifdef WARPX_QED
    Print() << "PICSAR (" << WarpX::PicsarVersion() << ")\n";
#endif

    if (restart_chkfile.empty())
    {
        ComputeDt();
        InitFromScratch();
    }
    else
    {
        InitFromCheckpoint();
        if (is_synchronized) {
            ComputeDt();
        }
        PostRestart();
    }

#ifdef AMREX_USE_EB
    ComputeEdgeLengths();
    ComputeFaceAreas();
    ScaleEdges();
    ScaleAreas();
#endif
    ComputeMaxStep();

    ComputePMLFactors();

    if (WarpX::use_fdtd_nci_corr) {
        WarpX::InitNCICorrector();
    }

    if (WarpX::use_filter) {
        WarpX::InitFilter();
    }

    BuildBufferMasks();

    if (WarpX::em_solver_medium==1) {
        m_macroscopic_properties->InitData();
    }

    InitDiagnostics();

    if (ParallelDescriptor::IOProcessor()) {
        std::cout << "\nGrids Summary:\n";
        printGridSummary(std::cout, 0, finestLevel());
    }

    // Check that the number of guard cells is smaller than the number of valid cells for all MultiFabs
    // (example: a box with 16 valid cells and 32 guard cells in z will not be considered valid)
    CheckGuardCells();

    if (restart_chkfile.empty())
    {
        multi_diags->FilterComputePackFlush( -1, true );

        // Write reduced diagnostics before the first iteration.
        if (reduced_diags->m_plot_rd != 0)
        {
            reduced_diags->ComputeDiags(-1);
            reduced_diags->WriteToFile(-1);
        }
    }

    PerformanceHints();
}

void
WarpX::InitDiagnostics () {
    multi_diags->InitData();
    if (do_back_transformed_diagnostics) {
        const Real* current_lo = geom[0].ProbLo();
        const Real* current_hi = geom[0].ProbHi();
        Real dt_boost = dt[0];
        // Find the positions of the lab-frame box that corresponds to the boosted-frame box at t=0
        Real zmin_lab = static_cast<Real>(
            current_lo[moving_window_dir]/( (1.+beta_boost)*gamma_boost ));
        Real zmax_lab = static_cast<Real>(
            current_hi[moving_window_dir]/( (1.+beta_boost)*gamma_boost ));
        myBFD = std::make_unique<BackTransformedDiagnostic>(
                                               zmin_lab,
                                               zmax_lab,
                                               moving_window_v, dt_snapshots_lab,
                                               num_snapshots_lab,
                                               dt_slice_snapshots_lab,
                                               num_slice_snapshots_lab,
                                               gamma_boost, t_new[0], dt_boost,
                                               moving_window_dir, geom[0],
                                               slice_realbox,
                                               particle_slice_width_lab);
    }
}

void
WarpX::InitFromScratch ()
{
    const Real time = 0.0;

    AmrCore::InitFromScratch(time);  // This will call MakeNewLevelFromScratch

    mypc->AllocData();
    mypc->InitData();

    // Loop through species and calculate their space-charge field
    bool const reset_fields = false; // Do not erase previous user-specified values on the grid
    ComputeSpaceChargeField(reset_fields);

    InitPML();
}

void
WarpX::InitPML ()
{

    // if periodicity defined in input, use existing pml interface
    amrex::Vector<int> geom_periodicity(AMREX_SPACEDIM,0);
    ParmParse pp_geometry("geometry");
    if (pp_geometry.queryarr("is_periodic", geom_periodicity)) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (geom_periodicity[idim] == 1) {
                do_pml_Lo[idim] = 0;
                do_pml_Hi[idim] = 0;
            }
        }
    } else {
        // setting do_pml = 0 as default and turning it on only when user-input is set to PML.
        do_pml = 0;
        do_pml_Lo = amrex::IntVect::TheZeroVector();
        do_pml_Hi = amrex::IntVect::TheZeroVector();
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (WarpX::field_boundary_lo[idim] == FieldBoundaryType::PML) {
                do_pml = 1;
                do_pml_Lo[idim] = 1;
            }
            if (WarpX::field_boundary_hi[idim] == FieldBoundaryType::PML) {
                do_pml = 1;
                do_pml_Hi[idim] = 1;
            }
        }
    }
    if (finest_level > 0) do_pml = 1;
    if (do_pml)
    {
        amrex::IntVect do_pml_Lo_corrected = do_pml_Lo;

#ifdef WARPX_DIM_RZ
        do_pml_Lo_corrected[0] = 0; // no PML at r=0, in cylindrical geometry
#endif
        pml[0] = std::make_unique<PML>(0, boxArray(0), DistributionMap(0), &Geom(0), nullptr,
                             pml_ncell, pml_delta, amrex::IntVect::TheZeroVector(),
                             dt[0], nox_fft, noy_fft, noz_fft, do_nodal,
                             do_moving_window, pml_has_particles, do_pml_in_domain,
                             do_pml_dive_cleaning, do_pml_divb_cleaning,
                             do_pml_Lo_corrected, do_pml_Hi);
        for (int lev = 1; lev <= finest_level; ++lev)
        {
            amrex::IntVect do_pml_Lo_MR = amrex::IntVect::TheUnitVector();
            amrex::IntVect do_pml_Hi_MR = amrex::IntVect::TheUnitVector();
            // check if fine patch edges co-incide with domain boundary
            amrex::Box levelBox = boxArray(lev).minimalBox();
            // Domain box at level, lev
            amrex::Box DomainBox = Geom(lev).Domain();
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                if (levelBox.smallEnd(idim) == DomainBox.smallEnd(idim))
                    do_pml_Lo_MR[idim] = do_pml_Lo[idim];
                if (levelBox.bigEnd(idim) == DomainBox.bigEnd(idim))
                    do_pml_Hi_MR[idim] = do_pml_Hi[idim];
            }

#ifdef WARPX_DIM_RZ
            //In cylindrical geometry, if the edge of the patch is at r=0, do not add PML
            if ((max_level > 0) && (fine_tag_lo[0]==0.)) {
                do_pml_Lo_MR[0] = 0;
            }
#endif
            pml[lev] = std::make_unique<PML>(lev, boxArray(lev), DistributionMap(lev),
                                   &Geom(lev), &Geom(lev-1),
                                   pml_ncell, pml_delta, refRatio(lev-1),
                                   dt[lev], nox_fft, noy_fft, noz_fft, do_nodal,
                                   do_moving_window, pml_has_particles, do_pml_in_domain,
                                   do_pml_dive_cleaning, do_pml_divb_cleaning,
                                   do_pml_Lo_MR, do_pml_Hi_MR);
        }
    }
}

void
WarpX::ComputePMLFactors ()
{
    if (do_pml)
    {
        for (int lev = 0; lev <= finest_level; ++lev)
        {
            pml[lev]->ComputePMLFactors(dt[lev]);
        }
    }
}

void
WarpX::ComputeMaxStep ()
{
    if (do_compute_max_step_from_zmax) {
        computeMaxStepBoostAccelerator(geom[0]);
    }

    // Make max_step and stop_time self-consistent, assuming constant dt.

    // If max_step is the limiting condition, decrease stop_time consistently
    if (stop_time > t_new[0] + dt[0]*(max_step - istep[0]) ) {
        stop_time = t_new[0] + dt[0]*(max_step - istep[0]);
    }
    // If stop_time is the limiting condition instead, decrease max_step consistently
    else {
        // The static_cast should not overflow since stop_time is the limiting condition here
        max_step = static_cast<int>(istep[0] + std::ceil( (stop_time-t_new[0])/dt[0] ));
    }
}

/* \brief computes max_step for wakefield simulation in boosted frame.
 * \param geom: Geometry object that contains simulation domain.
 *
 * max_step is set so that the simulation stop when the lower corner of the
 * simulation box passes input parameter zmax_plasma_to_compute_max_step.
 */
void
WarpX::computeMaxStepBoostAccelerator(const amrex::Geometry& a_geom){
    // Sanity checks: can use zmax_plasma_to_compute_max_step only if
    // the moving window and the boost are all in z direction.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::moving_window_dir == AMREX_SPACEDIM-1,
        "Can use zmax_plasma_to_compute_max_step only if " +
        "moving window along z. TODO: all directions.");
    if (gamma_boost > 1){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            (WarpX::boost_direction[0]-0)*(WarpX::boost_direction[0]-0) +
            (WarpX::boost_direction[1]-0)*(WarpX::boost_direction[1]-0) +
            (WarpX::boost_direction[2]-1)*(WarpX::boost_direction[2]-1) < 1.e-12,
            "Can use zmax_plasma_to_compute_max_step in boosted frame only if " +
            "warpx.boost_direction = z. TODO: all directions.");
    }

    // Lower end of the simulation domain. All quantities are given in boosted
    // frame except zmax_plasma_to_compute_max_step.
    const Real zmin_domain_boost = a_geom.ProbLo(AMREX_SPACEDIM-1);
    // End of the plasma: Transform input argument
    // zmax_plasma_to_compute_max_step to boosted frame.
    const Real len_plasma_boost = zmax_plasma_to_compute_max_step/gamma_boost;
    // Plasma velocity
    const Real v_plasma_boost = -beta_boost * PhysConst::c;
    // Get time at which the lower end of the simulation domain passes the
    // upper end of the plasma (in the z direction).
    const Real interaction_time_boost = (len_plasma_boost-zmin_domain_boost)/
        (moving_window_v-v_plasma_boost);
    // Divide by dt, and update value of max_step.
    int computed_max_step;
    if (do_subcycling){
        computed_max_step = static_cast<int>(interaction_time_boost/dt[0]);
    } else {
        computed_max_step =
            static_cast<int>(interaction_time_boost/dt[maxLevel()]);
    }
    max_step = computed_max_step;
    Print()<<"max_step computed in computeMaxStepBoostAccelerator: "
           <<computed_max_step<<std::endl;
}

void
WarpX::InitNCICorrector ()
{
    if (WarpX::use_fdtd_nci_corr)
    {
        for (int lev = 0; lev <= max_level; ++lev)
        {
            const Geometry& gm = Geom(lev);
            const Real* dx = gm.CellSize();
            amrex::Real dz, cdtodz;
            if (AMREX_SPACEDIM == 3){
                dz = dx[2];
            }else{
                dz = dx[1];
            }
            cdtodz = PhysConst::c * dt[lev] / dz;

            // Initialize Godfrey filters
            // Same filter for fields Ex, Ey and Bz
            const bool nodal_gather = !galerkin_interpolation;
            nci_godfrey_filter_exeybz[lev] = std::make_unique<NCIGodfreyFilter>(
                godfrey_coeff_set::Ex_Ey_Bz, cdtodz, nodal_gather);
            // Same filter for fields Bx, By and Ez
            nci_godfrey_filter_bxbyez[lev] = std::make_unique<NCIGodfreyFilter>(
                godfrey_coeff_set::Bx_By_Ez, cdtodz, nodal_gather);
            // Compute Godfrey filters stencils
            nci_godfrey_filter_exeybz[lev]->ComputeStencils();
            nci_godfrey_filter_bxbyez[lev]->ComputeStencils();
        }
    }
}

void
WarpX::InitFilter (){
    if (WarpX::use_filter){
        WarpX::bilinear_filter.npass_each_dir = WarpX::filter_npass_each_dir.toArray<unsigned int>();
        WarpX::bilinear_filter.ComputeStencils();
    }
}

void
WarpX::PostRestart ()
{
    if (WarpX::maxwell_solver_id == MaxwellSolverAlgo::PSATD) {
        amrex::Abort("WarpX::PostRestart: TODO for PSATD");
    }
    mypc->PostRestart();
}


void
WarpX::InitLevelData (int lev, Real /*time*/)
{

    ParmParse pp_warpx("warpx");

    // default values of E_external_grid and B_external_grid
    // are used to set the E and B field when "constant" or
    // "parser" is not explicitly used in the input.
    pp_warpx.query("B_ext_grid_init_style", B_ext_grid_s);
    std::transform(B_ext_grid_s.begin(),
                   B_ext_grid_s.end(),
                   B_ext_grid_s.begin(),
                   ::tolower);

#ifdef WARPX_MAG_LLG
    if (pp_warpx.query("B_ext_grid_init_style", B_ext_grid_s) ) {
        amrex::Abort("ERROR: Initialization of B field is not allowed in the LLG simulation! \nThe initial magnetic field must be H and M! \n");
    }
#endif

    pp_warpx.query("E_ext_grid_init_style", E_ext_grid_s);
    std::transform(E_ext_grid_s.begin(),
                   E_ext_grid_s.end(),
                   E_ext_grid_s.begin(),
                   ::tolower);
#ifdef WARPX_MAG_LLG
    pp_warpx.query("M_ext_grid_init_style", M_ext_grid_s); // user-defined initial M
    std::transform(M_ext_grid_s.begin(),
                   M_ext_grid_s.end(),
                   M_ext_grid_s.begin(),
                   ::tolower);

    pp_warpx.query("H_ext_grid_init_style", H_ext_grid_s); // user-defined initial H
    std::transform(H_ext_grid_s.begin(),
                   H_ext_grid_s.end(),
                   H_ext_grid_s.begin(),
                   ::tolower);

    pp_warpx.query("H_bias_ext_grid_init_style", H_bias_ext_grid_s); // user-defined initial M
    std::transform(H_bias_ext_grid_s.begin(),
                   H_bias_ext_grid_s.end(),
                   H_bias_ext_grid_s.begin(),
                   ::tolower);
#endif

    // Query for type of external space-time (xt) varying excitation
    pp_warpx.query("B_excitation_on_grid_style", B_excitation_grid_s);
    std::transform(B_excitation_grid_s.begin(),
               B_excitation_grid_s.end(),
               B_excitation_grid_s.begin(),
               ::tolower);

#ifdef WARPX_MAG_LLG
    if (pp_warpx.query("B_excitation_on_grid_style", B_excitation_grid_s)) {
        amrex::Abort("ERROR: Excitation of B field is not allowed in the LLG simulation! \nThe excited magnetic field must be H field! \n");
    }
#endif

    pp_warpx.query("E_excitation_on_grid_style", E_excitation_grid_s);
    std::transform(E_excitation_grid_s.begin(),
                   E_excitation_grid_s.end(),
                   E_excitation_grid_s.begin(),
                   ::tolower);

#ifdef WARPX_MAG_LLG
    pp_warpx.query("H_excitation_on_grid_style", H_excitation_grid_s);
    std::transform(H_excitation_grid_s.begin(),
                   H_excitation_grid_s.end(),
                   H_excitation_grid_s.begin(),
                   ::tolower);
#endif
    if (E_excitation_grid_s == "parse_e_excitation_grid_function") {
        // if E excitation type is set to parser then the corresponding
        // source type (hard=1, soft=2) must be specified for all components
        // using the flag function. Note that a flag value of 0 will not update
        // the field with the excitation.
        Store_parserString(pp_warpx, "Ex_excitation_flag_function(x,y,z)",
                                str_Ex_excitation_flag_function);
        Store_parserString(pp_warpx, "Ey_excitation_flag_function(x,y,z)",
                                str_Ey_excitation_flag_function);
        Store_parserString(pp_warpx, "Ez_excitation_flag_function(x,y,z)",
                                str_Ez_excitation_flag_function);
        Exfield_flag_parser.reset(new ParserWrapper<3>(
                   makeParser(str_Ex_excitation_flag_function,{"x","y","z"})));
        Eyfield_flag_parser.reset(new ParserWrapper<3>(
                   makeParser(str_Ey_excitation_flag_function,{"x","y","z"})));
        Ezfield_flag_parser.reset(new ParserWrapper<3>(
                   makeParser(str_Ez_excitation_flag_function,{"x","y","z"})));
    }
    if (B_excitation_grid_s == "parse_b_excitation_grid_function") {
        // if B excitation type is set to parser then the corresponding
        // source type (hard=1, soft=2) must be specified for all components
        // using the flag function. Note that a flag value of 0 will not update
        // the field with the excitation.
        Store_parserString(pp_warpx, "Bx_excitation_flag_function(x,y,z)",
                                str_Bx_excitation_flag_function);
        Store_parserString(pp_warpx, "By_excitation_flag_function(x,y,z)",
                                str_By_excitation_flag_function);
        Store_parserString(pp_warpx, "Bz_excitation_flag_function(x,y,z)",
                                str_Bz_excitation_flag_function);
        Bxfield_flag_parser.reset(new ParserWrapper<3>(
                   makeParser(str_Bx_excitation_flag_function,{"x","y","z"})));
        Byfield_flag_parser.reset(new ParserWrapper<3>(
                   makeParser(str_By_excitation_flag_function,{"x","y","z"})));
        Bzfield_flag_parser.reset(new ParserWrapper<3>(
                   makeParser(str_Bz_excitation_flag_function,{"x","y","z"})));
    }

#ifdef WARPX_MAG_LLG
    if (H_excitation_grid_s == "parse_h_excitation_grid_function") {
        // if H excitation type is set to parser then the corresponding
        // source type (hard=1, soft=2) must be specified for all components
        // using the flag function. Note that a flag value of 0 will not update
        // the field with the excitation.
        Store_parserString(pp_warpx, "Hx_excitation_flag_function(x,y,z)",
                                str_Hx_excitation_flag_function);
        Store_parserString(pp_warpx, "Hy_excitation_flag_function(x,y,z)",
                                str_Hy_excitation_flag_function);
        Store_parserString(pp_warpx, "Hz_excitation_flag_function(x,y,z)",
                                str_Hz_excitation_flag_function);
        Hxfield_flag_parser.reset(new ParserWrapper<3>(
                   makeParser(str_Hx_excitation_flag_function,{"x","y","z"})));
        Hyfield_flag_parser.reset(new ParserWrapper<3>(
                   makeParser(str_Hy_excitation_flag_function,{"x","y","z"})));
        Hzfield_flag_parser.reset(new ParserWrapper<3>(
                   makeParser(str_Hz_excitation_flag_function,{"x","y","z"})));
    }
#endif
    // * Functions with the string "arr" in their names get an Array of
    //   values from the given entry in the table.  The array argument is
    //   resized (if necessary) to hold all the values requested.
    //
    // * Functions without the string "arr" in their names get single
    //   values from the given entry in the table.

    // if the input string is "constant", the values for the
    // external grid must be provided in the input.
    if (B_ext_grid_s == "constant")
        getArrWithParser(pp_warpx, "B_external_grid", B_external_grid);

    // if the input string is "constant", the values for the
    // external grid must be provided in the input.
    if (E_ext_grid_s == "constant")
        getArrWithParser(pp_warpx, "E_external_grid", E_external_grid);

    // make parser for the external B-excitation in space-time
    if (B_excitation_grid_s == "parse_b_excitation_grid_function") {
#ifdef WARPX_DIM_RZ
       amrex::Abort("E and B parser for external fields does not work with RZ -- TO DO");
#endif
       Store_parserString(pp_warpx, "Bx_excitation_grid_function(x,y,z,t)",
                                                    str_Bx_excitation_grid_function);
       Store_parserString(pp_warpx, "By_excitation_grid_function(x,y,z,t)",
                                                    str_By_excitation_grid_function);
       Store_parserString(pp_warpx, "Bz_excitation_grid_function(x,y,z,t)",
                                                    str_Bz_excitation_grid_function);
       Bxfield_xt_grid_parser.reset(new ParserWrapper<4>(
                   makeParser(str_Bx_excitation_grid_function,{"x","y","z","t"})));
       Byfield_xt_grid_parser.reset(new ParserWrapper<4>(
                   makeParser(str_By_excitation_grid_function,{"x","y","z","t"})));
       Bzfield_xt_grid_parser.reset(new ParserWrapper<4>(
                   makeParser(str_Bz_excitation_grid_function,{"x","y","z","t"})));
    }

    // make parser for the external E-excitation in space-time
    if (E_excitation_grid_s == "parse_e_excitation_grid_function") {
#ifdef WARPX_DIM_RZ
       amrex::Abort("E and B parser for external fields does not work with RZ -- TO DO");
#endif
       Store_parserString(pp_warpx, "Ex_excitation_grid_function(x,y,z,t)",
                                                    str_Ex_excitation_grid_function);
       Store_parserString(pp_warpx, "Ey_excitation_grid_function(x,y,z,t)",
                                                    str_Ey_excitation_grid_function);
       Store_parserString(pp_warpx, "Ez_excitation_grid_function(x,y,z,t)",
                                                    str_Ez_excitation_grid_function);
       Exfield_xt_grid_parser.reset(new ParserWrapper<4>(
                   makeParser(str_Ex_excitation_grid_function,{"x","y","z","t"})));
       Eyfield_xt_grid_parser.reset(new ParserWrapper<4>(
                   makeParser(str_Ey_excitation_grid_function,{"x","y","z","t"})));
       Ezfield_xt_grid_parser.reset(new ParserWrapper<4>(
                   makeParser(str_Ez_excitation_grid_function,{"x","y","z","t"})));
    }

#ifdef WARPX_MAG_LLG
    // make parser for the external H-excitation in space-time
    if (H_excitation_grid_s == "parse_h_excitation_grid_function") {
#ifdef WARPX_DIM_RZ
       amrex::Abort("H parser for external fields does not work with RZ -- TO DO");
#endif
       Store_parserString(pp_warpx, "Hx_excitation_grid_function(x,y,z,t)",
                                                    str_Hx_excitation_grid_function);
       Store_parserString(pp_warpx, "Hy_excitation_grid_function(x,y,z,t)",
                                                    str_Hy_excitation_grid_function);
       Store_parserString(pp_warpx, "Hz_excitation_grid_function(x,y,z,t)",
                                                    str_Hz_excitation_grid_function);
       Hxfield_xt_grid_parser.reset(new ParserWrapper<4>(
                   makeParser(str_Hx_excitation_grid_function,{"x","y","z","t"})));
       Hyfield_xt_grid_parser.reset(new ParserWrapper<4>(
                   makeParser(str_Hy_excitation_grid_function,{"x","y","z","t"})));
       Hzfield_xt_grid_parser.reset(new ParserWrapper<4>(
                   makeParser(str_Hz_excitation_grid_function,{"x","y","z","t"})));
    }
#endif

#ifdef WARPX_MAG_LLG
    if (M_ext_grid_s == "constant")
        pp_warpx.getarr("M_external_grid", M_external_grid);

    if (H_ext_grid_s == "constant")
        pp_warpx.getarr("H_external_grid", H_external_grid);

    if (H_bias_ext_grid_s == "constant")
        pp_warpx.getarr("H_bias_external_grid", H_bias_external_grid);
#endif
    // initialize the averaged fields only if the averaged algorithm
    // is activated ('psatd.do_time_averaging=1')
    ParmParse pp_psatd("psatd");
    pp_psatd.query("do_time_averaging", fft_do_time_averaging );

    for (int i = 0; i < 3; ++i) {
        current_fp[lev][i]->setVal(0.0);
        if (lev > 0)
           current_cp[lev][i]->setVal(0.0);

        // Initialize aux MultiFabs on level 0
        if (lev == 0) {
            Bfield_aux[lev][i]->setVal(0.0);
            Efield_aux[lev][i]->setVal(0.0);
        }

        if (WarpX::do_current_centering)
        {
            current_fp_nodal[lev][i]->setVal(0.0);
        }

        if (B_ext_grid_s == "constant" || B_ext_grid_s == "default") {
           Bfield_fp[lev][i]->setVal(B_external_grid[i]);
           if (fft_do_time_averaging) {
                Bfield_avg_fp[lev][i]->setVal(B_external_grid[i]);
           }

           if (lev > 0) {
              Bfield_aux[lev][i]->setVal(B_external_grid[i]);
              Bfield_cp[lev][i]->setVal(B_external_grid[i]);
              if (fft_do_time_averaging) {
                  Bfield_avg_cp[lev][i]->setVal(B_external_grid[i]);
              }
           }
        }
        if (E_ext_grid_s == "constant" || E_ext_grid_s == "default") {
           Efield_fp[lev][i]->setVal(E_external_grid[i]);
           if (fft_do_time_averaging) {
               Efield_avg_fp[lev][i]->setVal(E_external_grid[i]);
            }

           if (lev > 0) {
              Efield_aux[lev][i]->setVal(E_external_grid[i]);
              Efield_cp[lev][i]->setVal(E_external_grid[i]);
              if (fft_do_time_averaging) {
                  Efield_avg_cp[lev][i]->setVal(E_external_grid[i]);
              }
           }
        }

#ifdef WARPX_MAG_LLG
        if (M_ext_grid_s == "constant" || M_ext_grid_s == "default"){
            // this if condition finds out if the user-input is constant
            // if not, set initial value to default, default = 0.0

            // Set the value of num_comp components in the valid region of
            // each FAB in the FabArray, starting at component comp to val.
            // Also set the value of nghost boundary cells.
            // template <class F=FAB, class = typename std::enable_if<IsBaseFab<F>::value>::type >
            // void setVal (value_type val,
            //              int        comp,
            //              int        num_comp,
            //              int        nghost = 0);

            int nghost = 1;
            for (int icomp = 0; icomp < 3; ++icomp){ // icomp is the index of components at each i face
                Mfield_fp[lev][i]->setVal(M_external_grid[icomp], icomp, 1, nghost);
            }
        }

        if (H_ext_grid_s == "constant" || H_ext_grid_s == "default") {
           Hfield_fp[lev][i]->setVal(H_external_grid[i]);
           if (lev > 0) {
              Hfield_aux[lev][i]->setVal(H_external_grid[i]);
              Hfield_cp[lev][i]->setVal(H_external_grid[i]);
           }
        }

        if (H_bias_ext_grid_s == "constant" || H_bias_ext_grid_s == "default") {
           H_biasfield_fp[lev][i]->setVal(H_bias_external_grid[i]);
           if (lev > 0) {
              H_biasfield_aux[lev][i]->setVal(H_bias_external_grid[i]);
              H_biasfield_cp[lev][i]->setVal(H_bias_external_grid[i]);
           }
        }

#endif

   }

    // if the input string for the B-field is "parse_b_ext_grid_function",
    // then the analytical expression or function must be
    // provided in the input file.
    if (B_ext_grid_s == "parse_b_ext_grid_function") {

#ifdef WARPX_DIM_RZ
       amrex::Abort("E and B parser for external fields does not work with RZ -- TO DO");
#endif
       Store_parserString(pp_warpx, "Bx_external_grid_function(x,y,z)",
                                                    str_Bx_ext_grid_function);
       Store_parserString(pp_warpx, "By_external_grid_function(x,y,z)",
                                                    str_By_ext_grid_function);
       Store_parserString(pp_warpx, "Bz_external_grid_function(x,y,z)",
                                                    str_Bz_ext_grid_function);
       Bxfield_parser = std::make_unique<ParserWrapper<3>>(
                                makeParser(str_Bx_ext_grid_function,{"x","y","z"}));
       Byfield_parser = std::make_unique<ParserWrapper<3>>(
                                makeParser(str_By_ext_grid_function,{"x","y","z"}));
       Bzfield_parser = std::make_unique<ParserWrapper<3>>(
                                makeParser(str_Bz_ext_grid_function,{"x","y","z"}));

       // Initialize Bfield_fp with external function
       InitializeExternalFieldsOnGridUsingParser(Bfield_fp[lev][0].get(),
                                                 Bfield_fp[lev][1].get(),
                                                 Bfield_fp[lev][2].get(),
                                                 getParser(Bxfield_parser),
                                                 getParser(Byfield_parser),
                                                 getParser(Bzfield_parser),
                                                 lev);
       if (lev > 0) {
          InitializeExternalFieldsOnGridUsingParser(Bfield_aux[lev][0].get(),
                                                    Bfield_aux[lev][1].get(),
                                                    Bfield_aux[lev][2].get(),
                                                    getParser(Bxfield_parser),
                                                    getParser(Byfield_parser),
                                                    getParser(Bzfield_parser),
                                                    lev);

          InitializeExternalFieldsOnGridUsingParser(Bfield_cp[lev][0].get(),
                                                    Bfield_cp[lev][1].get(),
                                                    Bfield_cp[lev][2].get(),
                                                    getParser(Bxfield_parser),
                                                    getParser(Byfield_parser),
                                                    getParser(Bzfield_parser),
                                                    lev);
       }
    }

    // if the input string for the E-field is "parse_e_ext_grid_function",
    // then the analytical expression or function must be
    // provided in the input file.
    if (E_ext_grid_s == "parse_e_ext_grid_function") {

#ifdef WARPX_DIM_RZ
       amrex::Abort("E and B parser for external fields does not work with RZ -- TO DO");
#endif
       Store_parserString(pp_warpx, "Ex_external_grid_function(x,y,z)",
                                                    str_Ex_ext_grid_function);
       Store_parserString(pp_warpx, "Ey_external_grid_function(x,y,z)",
                                                    str_Ey_ext_grid_function);
       Store_parserString(pp_warpx, "Ez_external_grid_function(x,y,z)",
                                                    str_Ez_ext_grid_function);

       Exfield_parser = std::make_unique<ParserWrapper<3>>(
                                makeParser(str_Ex_ext_grid_function,{"x","y","z"}));
       Eyfield_parser = std::make_unique<ParserWrapper<3>>(
                                makeParser(str_Ey_ext_grid_function,{"x","y","z"}));
       Ezfield_parser = std::make_unique<ParserWrapper<3>>(
                                makeParser(str_Ez_ext_grid_function,{"x","y","z"}));

       // Initialize Efield_fp with external function
       InitializeExternalFieldsOnGridUsingParser(Efield_fp[lev][0].get(),
                                                 Efield_fp[lev][1].get(),
                                                 Efield_fp[lev][2].get(),
                                                 getParser(Exfield_parser),
                                                 getParser(Eyfield_parser),
                                                 getParser(Ezfield_parser),
                                                 lev);
       if (lev > 0) {
          InitializeExternalFieldsOnGridUsingParser(Efield_aux[lev][0].get(),
                                                    Efield_aux[lev][1].get(),
                                                    Efield_aux[lev][2].get(),
                                                    getParser(Exfield_parser),
                                                    getParser(Eyfield_parser),
                                                    getParser(Ezfield_parser),
                                                    lev);

          InitializeExternalFieldsOnGridUsingParser(Efield_cp[lev][0].get(),
                                                    Efield_cp[lev][1].get(),
                                                    Efield_cp[lev][2].get(),
                                                    getParser(Exfield_parser),
                                                    getParser(Eyfield_parser),
                                                    getParser(Ezfield_parser),
                                                    lev);
       }
    }

#ifdef WARPX_MAG_LLG
    // if the input string for the Hbias-field is "parse_h_bias_ext_grid_function",
    // then the analytical expression or function must be
    // provided in the input file.
    if (H_bias_ext_grid_s == "parse_h_bias_ext_grid_function") {

#ifdef WARPX_DIM_RZ
       amrex::Abort("H bias parser for external fields does not work with RZ -- TO DO");
#endif
       Store_parserString(pp_warpx, "Hx_bias_external_grid_function(x,y,z)",
                                                    str_Hx_bias_ext_grid_function);
       Store_parserString(pp_warpx, "Hy_bias_external_grid_function(x,y,z)",
                                                    str_Hy_bias_ext_grid_function);
       Store_parserString(pp_warpx, "Hz_bias_external_grid_function(x,y,z)",
                                                    str_Hz_bias_ext_grid_function);

       Hx_biasfield_parser.reset(new ParserWrapper<3>(
                                makeParser(str_Hx_bias_ext_grid_function,{"x","y","z"})));
       Hy_biasfield_parser.reset(new ParserWrapper<3>(
                                makeParser(str_Hy_bias_ext_grid_function,{"x","y","z"})));
       Hz_biasfield_parser.reset(new ParserWrapper<3>(
                                makeParser(str_Hz_bias_ext_grid_function,{"x","y","z"})));

       // Initialize Efield_fp with external function
       InitializeExternalFieldsOnGridUsingParser(H_biasfield_fp[lev][0].get(),
                                                 H_biasfield_fp[lev][1].get(),
                                                 H_biasfield_fp[lev][2].get(),
                                                 getParser(Hx_biasfield_parser),
                                                 getParser(Hy_biasfield_parser),
                                                 getParser(Hz_biasfield_parser),
                                                 lev);
       if (lev > 0) {
          InitializeExternalFieldsOnGridUsingParser(H_biasfield_aux[lev][0].get(),
                                                    H_biasfield_aux[lev][1].get(),
                                                    H_biasfield_aux[lev][2].get(),
                                                    getParser(Hx_biasfield_parser),
                                                    getParser(Hy_biasfield_parser),
                                                    getParser(Hz_biasfield_parser),
                                                    lev);

          InitializeExternalFieldsOnGridUsingParser(H_biasfield_cp[lev][0].get(),
                                                    H_biasfield_cp[lev][1].get(),
                                                    H_biasfield_cp[lev][2].get(),
                                                    getParser(Hx_biasfield_parser),
                                                    getParser(Hy_biasfield_parser),
                                                    getParser(Hz_biasfield_parser),
                                                    lev);
       }
    }

    if (H_ext_grid_s == "parse_h_ext_grid_function") {

#ifdef WARPX_DIM_RZ
       amrex::Abort("H parser for external fields does not work with RZ -- TO DO");
#endif
       Store_parserString(pp_warpx, "Hx_external_grid_function(x,y,z)",
                                                    str_Hx_ext_grid_function);
       Store_parserString(pp_warpx, "Hy_external_grid_function(x,y,z)",
                                                    str_Hy_ext_grid_function);
       Store_parserString(pp_warpx, "Hz_external_grid_function(x,y,z)",
                                                    str_Hz_ext_grid_function);

       Hxfield_parser.reset(new ParserWrapper<3>(
                                makeParser(str_Hx_ext_grid_function,{"x","y","z"})));
       Hyfield_parser.reset(new ParserWrapper<3>(
                                makeParser(str_Hy_ext_grid_function,{"x","y","z"})));
       Hzfield_parser.reset(new ParserWrapper<3>(
                                makeParser(str_Hz_ext_grid_function,{"x","y","z"})));

       // Initialize Hfield_fp with external function
       InitializeExternalFieldsOnGridUsingParser(Hfield_fp[lev][0].get(),
                                                 Hfield_fp[lev][1].get(),
                                                 Hfield_fp[lev][2].get(),
                                                 getParser(Hxfield_parser),
                                                 getParser(Hyfield_parser),
                                                 getParser(Hzfield_parser),
                                                 lev);
       if (lev > 0) {
          InitializeExternalFieldsOnGridUsingParser(Hfield_aux[lev][0].get(),
                                                    Hfield_aux[lev][1].get(),
                                                    Hfield_aux[lev][2].get(),
                                                    getParser(Hxfield_parser),
                                                    getParser(Hyfield_parser),
                                                    getParser(Hzfield_parser),
                                                    lev);

          InitializeExternalFieldsOnGridUsingParser(Hfield_cp[lev][0].get(),
                                                    Hfield_cp[lev][1].get(),
                                                    Hfield_cp[lev][2].get(),
                                                    getParser(Hxfield_parser),
                                                    getParser(Hyfield_parser),
                                                    getParser(Hzfield_parser),
                                                    lev);
       }
    }

    if (M_ext_grid_s == "parse_m_ext_grid_function") {
#ifdef WARPX_DIM_RZ
        amrex::Abort("M-field parser for external fields does not work with RZ");
#endif
        Store_parserString(pp_warpx, "Mx_external_grid_function(x,y,z)",
                                                    str_Mx_ext_grid_function);
        Store_parserString(pp_warpx, "My_external_grid_function(x,y,z)",
                                                    str_My_ext_grid_function);
        Store_parserString(pp_warpx, "Mz_external_grid_function(x,y,z)",
                                                    str_Mz_ext_grid_function);

        Mxfield_parser.reset(new ParserWrapper<3>(
                                 makeParser(str_Mx_ext_grid_function,{"x","y","z"})));
        Myfield_parser.reset(new ParserWrapper<3>(
                                 makeParser(str_My_ext_grid_function,{"x","y","z"})));
        Mzfield_parser.reset(new ParserWrapper<3>(
                                 makeParser(str_Mz_ext_grid_function,{"x","y","z"})));

        {   // use this brace so Mx, My, Mz go out of scope
            // we need 1 more ghost cell than Mfield_fp has because
            // we are averaging to faces, including the ghost faces
            BoxArray bx = Mfield_fp[lev][0]->boxArray();
            amrex::MultiFab Mx(bx.enclosedCells(), Mfield_fp[lev][0]->DistributionMap(), 1, Mfield_fp[lev][0]->nGrow()+1);
            amrex::MultiFab My(bx.enclosedCells(), Mfield_fp[lev][1]->DistributionMap(), 1, Mfield_fp[lev][1]->nGrow()+1);
            amrex::MultiFab Mz(bx.enclosedCells(), Mfield_fp[lev][2]->DistributionMap(), 1, Mfield_fp[lev][2]->nGrow()+1);

            // Initialize Mfield_fp with external function
            InitializeExternalFieldsOnGridUsingParser(&Mx,
                                                      &My,
                                                      &Mz,
                                                      getParser(Mxfield_parser),
                                                      getParser(Myfield_parser),
                                                      getParser(Mzfield_parser),
                                                      lev);

            AverageParsedMtoFaces(Mx,My,Mz,*Mfield_fp[lev][0],*Mfield_fp[lev][1],*Mfield_fp[lev][2]);
        }

        if (lev > 0) {
            {   // use this brace so Mx, My, Mz go out of scope
                // we need 1 more ghost cell than Mfield_fp has because
                // we are averaging to faces, including the ghost faces
                BoxArray bx = Mfield_aux[lev][0]->boxArray();
                amrex::MultiFab Mx(bx.enclosedCells(), Mfield_aux[lev][0]->DistributionMap(), 1, Mfield_aux[lev][0]->nGrow()+1);
                amrex::MultiFab My(bx.enclosedCells(), Mfield_aux[lev][1]->DistributionMap(), 1, Mfield_aux[lev][1]->nGrow()+1);
                amrex::MultiFab Mz(bx.enclosedCells(), Mfield_aux[lev][2]->DistributionMap(), 1, Mfield_aux[lev][2]->nGrow()+1);

                InitializeExternalFieldsOnGridUsingParser(&Mx,
                                                          &My,
                                                          &Mz,
                                                          getParser(Mxfield_parser),
                                                          getParser(Myfield_parser),
                                                          getParser(Mzfield_parser),
                                                          lev);

                AverageParsedMtoFaces(Mx,My,Mz,*Mfield_aux[lev][0],*Mfield_aux[lev][1],*Mfield_aux[lev][2]);
            }

            {   // use this brace so Mx, My, Mz go out of scope
                // we need 1 more ghost cell than Mfield_fp has because
                // we are averaging to faces, including the ghost faces
                BoxArray bx = Mfield_cp[lev][0]->boxArray();
                amrex::MultiFab Mx(bx.enclosedCells(), Mfield_cp[lev][0]->DistributionMap(), 1, Mfield_cp[lev][0]->nGrow()+1);
                amrex::MultiFab My(bx.enclosedCells(), Mfield_cp[lev][1]->DistributionMap(), 1, Mfield_cp[lev][1]->nGrow()+1);
                amrex::MultiFab Mz(bx.enclosedCells(), Mfield_cp[lev][2]->DistributionMap(), 1, Mfield_cp[lev][2]->nGrow()+1);

                InitializeExternalFieldsOnGridUsingParser(&Mx,
                                                          &My,
                                                          &Mz,
                                                          getParser(Mxfield_parser),
                                                          getParser(Myfield_parser),
                                                          getParser(Mzfield_parser),
                                                          lev);

                AverageParsedMtoFaces(Mx,My,Mz,*Mfield_cp[lev][0],*Mfield_cp[lev][1],*Mfield_cp[lev][2]);
            }
        }
    }

#endif //closes #ifdef WARPX_MAG_LLG

    if (F_fp[lev]) {
        F_fp[lev]->setVal(0.0);
    }

    if (G_fp[lev]) {
        G_fp[lev]->setVal(0.0);
    }

    if (rho_fp[lev]) {
        rho_fp[lev]->setVal(0.0);
    }

    if (F_cp[lev]) {
        F_cp[lev]->setVal(0.0);
    }

    if (G_cp[lev]) {
        G_cp[lev]->setVal(0.0);
    }

    if (rho_cp[lev]) {
        rho_cp[lev]->setVal(0.0);
    }

    if (costs[lev]) {
        for (int i : costs[lev]->IndexArray()) {
            (*costs[lev])[i] = 0.0;
            WarpX::setLoadBalanceEfficiency(lev, -1);
        }
    }
}

#ifdef WARPX_MAG_LLG
void WarpX::AverageParsedMtoFaces(MultiFab& Mx_cc,
                                  MultiFab& My_cc,
                                  MultiFab& Mz_cc,
                                  MultiFab& Mx_face,
                                  MultiFab& My_face,
                                  MultiFab& Mz_face)
{
    // average Mx, My, Mz to faces
    for (MFIter mfi(Mx_face, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const amrex::Box& tbx = mfi.tilebox( IntVect(1,0,0), Mx_face.nGrowVect() );
        const amrex::Box& tby = mfi.tilebox( IntVect(0,1,0), My_face.nGrowVect() );
        const amrex::Box& tbz = mfi.tilebox( IntVect(0,0,1), Mz_face.nGrowVect() );

        auto const& mx_cc = Mx_cc.array(mfi);
        auto const& my_cc = My_cc.array(mfi);
        auto const& mz_cc = Mz_cc.array(mfi);

        auto const& mx_face = Mx_face.array(mfi);
        auto const& my_face = My_face.array(mfi);
        auto const& mz_face = Mz_face.array(mfi);

        amrex::ParallelFor (tbx, tby, tbz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            mx_face(i,j,k,0) = 0.5*(mx_cc(i-1,j,k) + mx_cc(i,j,k));
            mx_face(i,j,k,1) = 0.5*(my_cc(i-1,j,k) + my_cc(i,j,k));
            mx_face(i,j,k,2) = 0.5*(mz_cc(i-1,j,k) + mz_cc(i,j,k));
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            my_face(i,j,k,0) = 0.5*(mx_cc(i,j-1,k) + mx_cc(i,j,k));
            my_face(i,j,k,1) = 0.5*(my_cc(i,j-1,k) + my_cc(i,j,k));
            my_face(i,j,k,2) = 0.5*(mz_cc(i,j-1,k) + mz_cc(i,j,k));
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            mz_face(i,j,k,0) = 0.5*(mx_cc(i,j,k-1) + mx_cc(i,j,k));
            mz_face(i,j,k,1) = 0.5*(my_cc(i,j,k-1) + my_cc(i,j,k));
            mz_face(i,j,k,2) = 0.5*(mz_cc(i,j,k-1) + mz_cc(i,j,k));
        });
    }
}
#endif

void
WarpX::InitializeExternalFieldsOnGridUsingParser (
       MultiFab *mfx, MultiFab *mfy, MultiFab *mfz,
       HostDeviceParser<3> const& xfield_parser, HostDeviceParser<3> const& yfield_parser,
       HostDeviceParser<3> const& zfield_parser, const int lev)
{
    const auto dx_lev = geom[lev].CellSizeArray();
    const RealBox& real_box = geom[lev].ProbDomain();
    amrex::IntVect x_nodal_flag = mfx->ixType().toIntVect();
    amrex::IntVect y_nodal_flag = mfy->ixType().toIntVect();
    amrex::IntVect z_nodal_flag = mfz->ixType().toIntVect();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        mfx->nComp() == mfy->nComp() and mfx->nComp() == mfz->nComp(),
        "The number of components for the three Multifabs must be equal");
    // Number of multifab components
    int ncomp = mfx->nComp();
    for ( MFIter mfi(*mfx, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {

       const amrex::Box& tbx = mfi.tilebox( x_nodal_flag, mfx->nGrowVect() );
       const amrex::Box& tby = mfi.tilebox( y_nodal_flag, mfy->nGrowVect() );
       const amrex::Box& tbz = mfi.tilebox( z_nodal_flag, mfz->nGrowVect() );

       auto const& mfxfab = mfx->array(mfi);
       auto const& mfyfab = mfy->array(mfi);
       auto const& mfzfab = mfz->array(mfi);
       amrex::ParallelFor (tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Shift required in the x-, y-, or z- position
                // depending on the index type of the multifab
                amrex::Real fac_x = (1._rt - x_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                amrex::Real y = 0._rt;
                amrex::Real fac_z = (1._rt - x_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real z = j*dx_lev[1] + real_box.lo(1) + fac_z;
#else
                amrex::Real fac_y = (1._rt - x_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real y = j*dx_lev[1] + real_box.lo(1) + fac_y;
                amrex::Real fac_z = (1._rt - x_nodal_flag[2]) * dx_lev[2] * 0.5_rt;
                amrex::Real z = k*dx_lev[2] + real_box.lo(2) + fac_z;
#endif
#ifdef WARPX_MAG_LLG
                if (ncomp > 1) {
                    // This condition is specific to Mfield, where,
                    // x-, y-, and z-components are stored on the x-face
                    mfxfab(i,j,k,0) = xfield_parser(x,y,z);
                    mfxfab(i,j,k,1) = yfield_parser(x,y,z);
                    mfxfab(i,j,k,2) = zfield_parser(x,y,z);
                } else
#endif
                {
                    mfxfab(i,j,k) = xfield_parser(x,y,z);
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real fac_x = (1._rt - y_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                amrex::Real y = 0._rt;
                amrex::Real fac_z = (1._rt - y_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real z = j*dx_lev[1] + real_box.lo(1) + fac_z;
#elif (AMREX_SPACEDIM==3)
                amrex::Real fac_y = (1._rt - y_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real y = j*dx_lev[1] + real_box.lo(1) + fac_y;
                amrex::Real fac_z = (1._rt - y_nodal_flag[2]) * dx_lev[2] * 0.5_rt;
                amrex::Real z = k*dx_lev[2] + real_box.lo(2) + fac_z;
#endif
#ifdef WARPX_MAG_LLG
                if (ncomp > 1) {
                    // This condition is specific to Mfield, where,
                    // x-, y-, and z-components are stored on the y-face
                    mfyfab(i,j,k,0) = xfield_parser(x,y,z);
                    mfyfab(i,j,k,1) = yfield_parser(x,y,z);
                    mfyfab(i,j,k,2) = zfield_parser(x,y,z);
                } else
#endif
                {
                    mfyfab(i,j,k)  = yfield_parser(x,y,z);
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real fac_x = (1._rt - z_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                amrex::Real y = 0._rt;
                amrex::Real fac_z = (1._rt - z_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real z = j*dx_lev[1] + real_box.lo(1) + fac_z;
#elif (AMREX_SPACEDIM==3)
                amrex::Real fac_y = (1._rt - z_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real y = j*dx_lev[1] + real_box.lo(1) + fac_y;
                amrex::Real fac_z = (1._rt - z_nodal_flag[2]) * dx_lev[2] * 0.5_rt;
                amrex::Real z = k*dx_lev[2] + real_box.lo(2) + fac_z;
#endif
#ifdef WARPX_MAG_LLG
                if (ncomp > 1) {
                    // This condition is specific to Mfield, where,
                    // x-, y-, and z-components are stored on the z-face
                    mfzfab(i,j,k,0) = xfield_parser(x,y,z);
                    mfzfab(i,j,k,1) = yfield_parser(x,y,z);
                    mfzfab(i,j,k,2) = zfield_parser(x,y,z);
                } else
#endif
                {
                    mfzfab(i,j,k) = zfield_parser(x,y,z);
                }
            }
        );
    }
}

void
WarpX::PerformanceHints ()
{
    // Check requested MPI ranks and available boxes
    amrex::Long total_nboxes = 0; // on all MPI ranks
    for (int ilev = 0; ilev <= finestLevel(); ++ilev) {
        total_nboxes += boxArray(ilev).size();
    }
    if (ParallelDescriptor::NProcs() > total_nboxes)
        amrex::Print() << "\n[Warning] [Performance] Too many resources / too little work!\n"
            << "  It looks like you requested more compute resources than "
            << "there are total number of boxes of cells available ("
            << total_nboxes << "). "
            << "You started with (" << ParallelDescriptor::NProcs()
            << ") MPI ranks, so (" << ParallelDescriptor::NProcs() - total_nboxes
            << ") rank(s) will have no work.\n"
#ifdef AMREX_USE_GPU
            << "  On GPUs, consider using 1-8 boxes per GPU that together fill "
            << "each GPU's memory sufficiently. If you do not rely on dynamic "
            << "load-balancing, then one large box per GPU is ideal.\n"
#endif
            << "  More information:\n"
            << "  https://warpx.readthedocs.io/en/latest/running_cpp/parallelization.html\n";

    // TODO: warn if some ranks have disproportionally more work than all others
    //       tricky: it can be ok to assign "vacuum" boxes to some ranks w/o slowing down
    //               all other ranks; we need to measure this with our load-balancing
    //               routines and issue a warning only of some ranks stall all other ranks
    // TODO: check MPI-rank to GPU ratio (should be 1:1)
    // TODO: check memory per MPI rank, especially if GPUs are underutilized
    // TODO: CPU tiling hints with OpenMP
}

void WarpX::CheckGuardCells()
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        for (int dim = 0; dim < 3; ++dim)
        {
            CheckGuardCells(*Efield_fp[lev][dim]);
            CheckGuardCells(*Bfield_fp[lev][dim]);
            CheckGuardCells(*current_fp[lev][dim]);

            if (WarpX::fft_do_time_averaging)
            {
                CheckGuardCells(*Efield_avg_fp[lev][dim]);
                CheckGuardCells(*Bfield_avg_fp[lev][dim]);
            }
        }

        if (rho_fp[lev])
        {
            CheckGuardCells(*rho_fp[lev]);
        }

        if (F_fp[lev])
        {
            CheckGuardCells(*F_fp[lev]);
        }

        // MultiFabs on coarse patch
        if (lev > 0)
        {
            for (int dim = 0; dim < 3; ++dim)
            {
                CheckGuardCells(*Efield_cp[lev][dim]);
                CheckGuardCells(*Bfield_cp[lev][dim]);
                CheckGuardCells(*current_cp[lev][dim]);

                if (WarpX::fft_do_time_averaging)
                {
                    CheckGuardCells(*Efield_avg_cp[lev][dim]);
                    CheckGuardCells(*Bfield_avg_cp[lev][dim]);
                }
            }

            if (rho_cp[lev])
            {
                CheckGuardCells(*rho_cp[lev]);
            }

            if (F_cp[lev])
            {
                CheckGuardCells(*F_cp[lev]);
            }
        }
    }
}

void WarpX::CheckGuardCells(amrex::MultiFab const& mf)
{
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::IntVect vc = mfi.validbox().enclosedCells().size();
        const amrex::IntVect gc = mf.nGrowVect();
        if (vc.allGT(gc) == false)
        {
            std::stringstream ss;
            ss << "\nMultiFab "
               << mf.tags()[1]
               << ":\nthe number of guard cells "
               << gc
               << " is larger than or equal to the number of valid cells "
               << vc
               << ",\nplease reduce the number of guard cells"
               << " or increase the grid size by changing domain decomposition";
            amrex::Abort(ss.str());
        }
    }
}
