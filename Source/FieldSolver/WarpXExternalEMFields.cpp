#include "WarpX.H"
#include "BoundaryConditions/PML.H"
#include "Evolve/WarpXDtType.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXUtil.H"
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>

using namespace amrex;

/**
 * \brief externalfieldtype determines which field component the external excitation is applied on
 * externalfieldtype == ExternalFieldType::AllExternal : external field excitation applied to all three field components, E, B and H
 * externalfieldtype == ExternalFieldType::EfieldExternal : external field excitation applied to E field only
 * externalfieldtype == ExternalFieldType::BfieldExternal : external field excitation applied to B field only
 * externalfieldtype == ExternalFieldType::HfieldExternal : external field excitation applied to H field only; this option is only valid when USE_LLG == TRUE
 */

void
WarpX::ApplyExternalFieldExcitationOnGrid (int const externalfieldtype, DtType a_dt_type)
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        if (externalfieldtype == ExternalFieldType::AllExternal || externalfieldtype == ExternalFieldType::EfieldExternal) {
            if (E_excitation_grid_s == "parse_e_excitation_grid_function") {
                ApplyExternalFieldExcitationOnGrid(Efield_fp[lev][0].get(),
                                                   Efield_fp[lev][1].get(),
                                                   Efield_fp[lev][2].get(),
                                                   Exfield_xt_grid_parser->compile<4>(),
                                                   Eyfield_xt_grid_parser->compile<4>(),
                                                   Ezfield_xt_grid_parser->compile<4>(),
                                                   Exfield_flag_parser->compile<3>(),
                                                   Eyfield_flag_parser->compile<3>(),
                                                   Ezfield_flag_parser->compile<3>(),
                                                   lev, a_dt_type );
            }
        }
        // The excitation, especially when used to set an internal PEC, will be extended
        // to the PML region with user-defined parser.
        // As clarified in the documentation, it is important that the parser is valid in the pml region
        if (WarpX::isAnyBoundaryPML() and externalfieldtype == ExternalFieldType::EfieldExternalPML) {
            if (E_excitation_grid_s == "parse_e_excitation_grid_function") {
                    ApplyExternalFieldExcitationOnGrid(pml[lev]->GetE_fp(0),
                                                       pml[lev]->GetE_fp(1),
                                                       pml[lev]->GetE_fp(2),
                                                       Exfield_xt_grid_parser->compile<4>(),
                                                       Eyfield_xt_grid_parser->compile<4>(),
                                                       Ezfield_xt_grid_parser->compile<4>(),
                                                       Exfield_flag_parser->compile<3>(),
                                                       Eyfield_flag_parser->compile<3>(),
                                                       Ezfield_flag_parser->compile<3>(),
                                                       lev, a_dt_type );
            }
        }
        if (externalfieldtype == ExternalFieldType::AllExternal || externalfieldtype == ExternalFieldType::BfieldExternal) {
            if (B_excitation_grid_s == "parse_b_excitation_grid_function") {
                ApplyExternalFieldExcitationOnGrid(Bfield_fp[lev][0].get(),
                                                   Bfield_fp[lev][1].get(),
                                                   Bfield_fp[lev][2].get(),
                                                   Bxfield_xt_grid_parser->compile<4>(),
                                                   Byfield_xt_grid_parser->compile<4>(),
                                                   Bzfield_xt_grid_parser->compile<4>(),
                                                   Bxfield_flag_parser->compile<3>(),
                                                   Byfield_flag_parser->compile<3>(),
                                                   Bzfield_flag_parser->compile<3>(),
                                                   lev, a_dt_type );
            }
        }
#ifdef WARPX_MAG_LLG
        if (externalfieldtype == ExternalFieldType::AllExternal || externalfieldtype == ExternalFieldType::HfieldExternal) {
            if (H_excitation_grid_s == "parse_h_excitation_grid_function") {
            ApplyExternalFieldExcitationOnGrid(Hfield_fp[lev][0].get(),
                                               Hfield_fp[lev][1].get(),
                                               Hfield_fp[lev][2].get(),
                                               Hxfield_xt_grid_parser->compile<4>(),
                                               Hyfield_xt_grid_parser->compile<4>(),
                                               Hzfield_xt_grid_parser->compile<4>(),
                                               Hxfield_flag_parser->compile<3>(),
                                               Hyfield_flag_parser->compile<3>(),
                                               Hzfield_flag_parser->compile<3>(),
                                               lev, a_dt_type );
            }
        }
        if (externalfieldtype == ExternalFieldType::AllExternal || externalfieldtype == ExternalFieldType::HbiasfieldExternal) {
            if (H_bias_excitation_grid_s == "parse_h_bias_excitation_grid_function") {
            ApplyExternalFieldExcitationOnGrid(H_biasfield_fp[lev][0].get(),
                                               H_biasfield_fp[lev][1].get(),
                                               H_biasfield_fp[lev][2].get(),
                                               Hx_biasfield_xt_grid_parser->compile<4>(),
                                               Hy_biasfield_xt_grid_parser->compile<4>(),
                                               Hz_biasfield_xt_grid_parser->compile<4>(),
                                               Hx_biasfield_flag_parser->compile<3>(),
                                               Hy_biasfield_flag_parser->compile<3>(),
                                               Hz_biasfield_flag_parser->compile<3>(),
                                               lev, a_dt_type );
            }
        }
#endif
    } // for loop over level
}

void
WarpX::ApplyExternalFieldExcitationOnGrid (
       amrex::MultiFab *mfx, amrex::MultiFab *mfy, amrex::MultiFab *mfz,
       ParserExecutor<4> const& xfield_parser,
       ParserExecutor<4> const& yfield_parser,
       ParserExecutor<4> const& zfield_parser,
       ParserExecutor<3> const& xflag_parser,
       ParserExecutor<3> const& yflag_parser,
       ParserExecutor<3> const& zflag_parser, const int lev, DtType a_dt_type )
{
    // This function adds the contribution from an external excitation to the fields.
    // A flag is used to determine the type of excitation.
    // If flag == 1, it is a hard source and the field = excitation
    // If flag == 2, if is a soft source and the field += excitation
    // If flag == 0, the excitation parser is not computed and the field is unchanged.
    // If flag is not 0, or 1, or 2, the code will Abort!

    // Gpu vector to store Ex-Bz staggering (Hx-Hz for LLG)
    GpuArray<int,3> mfx_stag, mfy_stag, mfz_stag;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        mfx_stag[idim] = mfx->ixType()[idim];
        mfy_stag[idim] = mfy->ixType()[idim];
        mfz_stag[idim] = mfz->ixType()[idim];
    }
    amrex::Real t = gett_new(lev);
    const auto problo = Geom(lev).ProbLoArray();
    const auto dx = Geom(lev).CellSizeArray();
    amrex::IntVect x_nodal_flag = mfx->ixType().toIntVect();
    amrex::IntVect y_nodal_flag = mfy->ixType().toIntVect();
    amrex::IntVect z_nodal_flag = mfz->ixType().toIntVect();
    // For each multifab, apply excitation to ncomponents
    // If not split pml fields, the excitation is applied to the regular Efield used in Maxwell's eq.
    // If pml field, then the excitation is applied to all the split field components.
    const int nComp_x = mfx->nComp();
    const int nComp_y = mfy->nComp();
    const int nComp_z = mfz->nComp();
    // Multiplication factor for field parser depending on dt_type
    // If Full, then 1 (default), if FirstHalf or SecondHalf then 0.5
    int dt_type_flag = 0;
    if (a_dt_type == DtType::FirstHalf or a_dt_type == DtType::SecondHalf ) {
        dt_type_flag = 1;
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*mfx, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // Extract field data for this grid/tile
        amrex::Array4<amrex::Real> const& Fx = mfx->array(mfi);
        amrex::Array4<amrex::Real> const& Fy = mfy->array(mfi);
        amrex::Array4<amrex::Real> const& Fz = mfz->array(mfi);

        const amrex::Box& tbx = mfi.tilebox( x_nodal_flag, mfx->nGrowVect() );
        const amrex::Box& tby = mfi.tilebox( y_nodal_flag, mfy->nGrowVect() );
        const amrex::Box& tbz = mfi.tilebox( z_nodal_flag, mfz->nGrowVect() );

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, nComp_x,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfx_stag,
                                                  problo, dx, x, y, z);
                auto flag_type = xflag_parser(x,y,z);
                amrex::Real dt_type_factor = 1._rt;
                // For soft source and FirstHalf/SecondHalf evolve
                // the excitation is split with a prefector of 0.5
                if (flag_type == 2._rt and dt_type_flag == 1) {
                    dt_type_factor = 0.5_rt;
                }
                if (flag_type != 0._rt && flag_type != 1._rt && flag_type != 2._rt) {
                    amrex::Abort("flag type for excitation must be 0, or 1, or 2!");
                } else if ( flag_type > 0._rt ) {
                    Fx(i, j, k, n) = Fx(i,j,k,n)*(flag_type-1.0_rt)
                                   + dt_type_factor * xfield_parser(x,y,z,t);
                }
            },
            tby, nComp_y,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfy_stag,
                                                  problo, dx, x, y, z);
                auto flag_type = yflag_parser(x,y,z);
                amrex::Real dt_type_factor = 1._rt;
                // For soft source and FirstHalf/SecondHalf evolve
                // the excitation is split with a prefector of 0.5
                if (flag_type == 2._rt and dt_type_flag == 1) {
                    dt_type_factor = 0.5_rt;
                }
                if (flag_type != 0._rt && flag_type != 1._rt && flag_type != 2._rt) {
                    amrex::Abort("flag type for excitation must be 0, or 1, or 2!");
                } else if ( flag_type > 0._rt ) {
                    Fy(i, j, k, n) = Fy(i,j,k,n)*(flag_type-1.0_rt)
                                   + dt_type_factor * yfield_parser(x,y,z,t);
                }
            },
            tbz, nComp_z,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfz_stag,
                                                  problo, dx, x, y, z);
                auto flag_type = zflag_parser(x,y,z);
                amrex::Real dt_type_factor = 1._rt;
                // For soft source and FirstHalf/SecondHalf evolve
                // the excitation is split with a prefector of 0.5
                if (flag_type == 2._rt and dt_type_flag == 1) {
                    dt_type_factor = 0.5_rt;
                }
                if (flag_type != 0._rt && flag_type != 1._rt && flag_type != 2._rt) {
                    amrex::Abort("flag type for excitation must be 0, or 1, or 2!");
                } else if ( flag_type > 0._rt ) {
                    Fz(i, j, k,n) = Fz(i,j,k,n)*(flag_type-1.0_rt)
                                  + dt_type_factor * zfield_parser(x,y,z,t);
                }
            }
        );
    }
}

void
WarpX::ReadExcitationParser ()
{

    ParmParse pp_warpx("warpx");

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
    pp_warpx.query("H_bias_excitation_on_grid_style", H_bias_excitation_grid_s);
    std::transform(H_bias_excitation_grid_s.begin(),
                   H_bias_excitation_grid_s.end(),
                   H_bias_excitation_grid_s.begin(),
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
        Exfield_flag_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Ex_excitation_flag_function,{"x","y","z"}));
        Eyfield_flag_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Ey_excitation_flag_function,{"x","y","z"}));
        Ezfield_flag_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Ez_excitation_flag_function,{"x","y","z"}));

        pp_warpx.query("Apply_E_excitation_in_pml_region", ApplyExcitationInPML);
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
        Bxfield_flag_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Bx_excitation_flag_function,{"x","y","z"}));
        Byfield_flag_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_By_excitation_flag_function,{"x","y","z"}));
        Bzfield_flag_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Bz_excitation_flag_function,{"x","y","z"}));
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
        Hxfield_flag_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Hx_excitation_flag_function,{"x","y","z"}));
        Hyfield_flag_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Hy_excitation_flag_function,{"x","y","z"}));
        Hzfield_flag_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Hz_excitation_flag_function,{"x","y","z"}));
    }
    if (H_bias_excitation_grid_s == "parse_h_bias_excitation_grid_function") {
        // if H bias_excitation type is set to parser then the corresponding
        // source type (hard=1, soft=2) must be specified for all components
        // using the flag function. Note that a flag value of 0 will not update
        // the field with the excitation.
        Store_parserString(pp_warpx, "Hx_bias_excitation_flag_function(x,y,z)",
                                str_Hx_bias_excitation_flag_function);
        Store_parserString(pp_warpx, "Hy_bias_excitation_flag_function(x,y,z)",
                                str_Hy_bias_excitation_flag_function);
        Store_parserString(pp_warpx, "Hz_bias_excitation_flag_function(x,y,z)",
                                str_Hz_bias_excitation_flag_function);
        Hx_biasfield_flag_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Hx_bias_excitation_flag_function,{"x","y","z"}));
        Hy_biasfield_flag_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Hy_bias_excitation_flag_function,{"x","y","z"}));
        Hz_biasfield_flag_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Hz_bias_excitation_flag_function,{"x","y","z"}));
    }
#endif

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
       Bxfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Bx_excitation_grid_function,{"x","y","z","t"}));
       Byfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_By_excitation_grid_function,{"x","y","z","t"}));
       Bzfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Bz_excitation_grid_function,{"x","y","z","t"}));
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
       Exfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Ex_excitation_grid_function,{"x","y","z","t"}));
       Eyfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Ey_excitation_grid_function,{"x","y","z","t"}));
       Ezfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Ez_excitation_grid_function,{"x","y","z","t"}));
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
       Hxfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Hx_excitation_grid_function,{"x","y","z","t"}));
       Hyfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Hy_excitation_grid_function,{"x","y","z","t"}));
       Hzfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Hz_excitation_grid_function,{"x","y","z","t"}));
    }
    // make parser for the external H-biasexcitation in space-time
    if (H_bias_excitation_grid_s == "parse_h_bias_excitation_grid_function") {
#ifdef WARPX_DIM_RZ
       amrex::Abort("H parser for external fields does not work with RZ -- TO DO");
#endif
       Store_parserString(pp_warpx, "Hx_bias_excitation_grid_function(x,y,z,t)",
                                                    str_Hx_bias_excitation_grid_function);
       Store_parserString(pp_warpx, "Hy_bias_excitation_grid_function(x,y,z,t)",
                                                    str_Hy_bias_excitation_grid_function);
       Store_parserString(pp_warpx, "Hz_bias_excitation_grid_function(x,y,z,t)",
                                                    str_Hz_bias_excitation_grid_function);
       Hx_biasfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Hx_bias_excitation_grid_function,{"x","y","z","t"}));
       Hy_biasfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Hy_bias_excitation_grid_function,{"x","y","z","t"}));
       Hz_biasfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   makeParser(str_Hz_bias_excitation_grid_function,{"x","y","z","t"}));
    }
#endif
}
