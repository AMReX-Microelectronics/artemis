#include "WarpX.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXUtil.H"
#include "Parser/WarpXParserWrapper.H"
#include "Parser/GpuParser.H"

using namespace amrex;

/**
 * \brief externalfieldtype determines which field component the external excitation is applied on
 * externalfieldtype == ExternalFieldType::AllExternal : external field excitation applied to all three field components, E, B and H
 * externalfieldtype == ExternalFieldType::EfieldExternal : external field excitation applied to E field only
 * externalfieldtype == ExternalFieldType::BfieldExternal : external field excitation applied to B field only
 * externalfieldtype == ExternalFieldType::HfieldExternal : external field excitation applied to H field only; this option is only valid when USE_LLG == TRUE
 */

void
WarpX::ApplyExternalFieldExcitationOnGrid (int const externalfieldtype)
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        if (externalfieldtype == ExternalFieldType::AllExternal || externalfieldtype == ExternalFieldType::EfieldExternal) {
            if (E_excitation_grid_s == "parse_e_excitation_grid_function") {
                ApplyExternalFieldExcitationOnGrid(Efield_fp[lev][0].get(),
                                                   Efield_fp[lev][1].get(),
                                                   Efield_fp[lev][2].get(),
                                                   getParser(Exfield_xt_grid_parser),
                                                   getParser(Eyfield_xt_grid_parser),
                                                   getParser(Ezfield_xt_grid_parser),
                                                   getParser(Exfield_flag_parser),
                                                   getParser(Eyfield_flag_parser),
                                                   getParser(Ezfield_flag_parser),
                                                   lev );
            }
        }
        if (externalfieldtype == ExternalFieldType::AllExternal || externalfieldtype == ExternalFieldType::BfieldExternal) {
            if (B_excitation_grid_s == "parse_b_excitation_grid_function") {
                ApplyExternalFieldExcitationOnGrid(Bfield_fp[lev][0].get(),
                                                   Bfield_fp[lev][1].get(),
                                                   Bfield_fp[lev][2].get(),
                                                   getParser(Bxfield_xt_grid_parser),
                                                   getParser(Byfield_xt_grid_parser),
                                                   getParser(Bzfield_xt_grid_parser),
                                                   getParser(Bxfield_flag_parser),
                                                   getParser(Byfield_flag_parser),
                                                   getParser(Bzfield_flag_parser),
                                                   lev );
            }
        }
#ifdef WARPX_MAG_LLG
        if (externalfieldtype == ExternalFieldType::AllExternal || externalfieldtype == ExternalFieldType::HfieldExternal) {
            if (H_excitation_grid_s == "parse_h_excitation_grid_function") {
            ApplyExternalFieldExcitationOnGrid(Hfield_fp[lev][0].get(),
                                               Hfield_fp[lev][1].get(),
                                               Hfield_fp[lev][2].get(),
                                               getParser(Hxfield_xt_grid_parser),
                                               getParser(Hyfield_xt_grid_parser),
                                               getParser(Hzfield_xt_grid_parser),
                                               getParser(Hxfield_flag_parser),
                                               getParser(Hyfield_flag_parser),
                                               getParser(Hzfield_flag_parser),
                                               lev );
            }
        }
#endif
    } // for loop over level
}

void
WarpX::ApplyExternalFieldExcitationOnGrid (
       amrex::MultiFab *mfx, amrex::MultiFab *mfy, amrex::MultiFab *mfz,
       HostDeviceParser<4> const& xfield_parser,
       HostDeviceParser<4> const& yfield_parser,
       HostDeviceParser<4> const& zfield_parser,
       HostDeviceParser<3> const& xflag_parser,
       HostDeviceParser<3> const& yflag_parser,
       HostDeviceParser<3> const& zflag_parser, const int lev )
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
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*mfx, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // Extract field data for this grid/tile
        amrex::Array4<amrex::Real> const& Fx = mfx->array(mfi);
        amrex::Array4<amrex::Real> const& Fy = mfy->array(mfi);
        amrex::Array4<amrex::Real> const& Fz = mfz->array(mfi);

        const amrex::Box& tbx = mfi.tilebox( mfx->ixType().toIntVect() );
        const amrex::Box& tby = mfi.tilebox( mfy->ixType().toIntVect() );
        const amrex::Box& tbz = mfi.tilebox( mfz->ixType().toIntVect() );

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfx_stag,
                                                  problo, dx, x, y, z);
                auto flag_type = xflag_parser(x,y,z);
                if (flag_type != 0._rt && flag_type != 1._rt && flag_type != 2._rt) {
                    amrex::Abort("flag type for excitation must be 0, or 1, or 2!");
                } else if ( flag_type > 0._rt ) {
                    Fx(i, j, k) = Fx(i,j,k)*(flag_type-1.0_rt) + xfield_parser(x,y,z,t);
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfy_stag,
                                                  problo, dx, x, y, z);
                auto flag_type = yflag_parser(x,y,z);
                if (flag_type != 0._rt && flag_type != 1._rt && flag_type != 2._rt) {
                    amrex::Abort("flag type for excitation must be 0, or 1, or 2!");
                } else if ( flag_type > 0._rt ) {
                    Fy(i, j, k) = Fy(i,j,k)*(flag_type-1.0_rt) + yfield_parser(x,y,z,t);
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfz_stag,
                                                  problo, dx, x, y, z);
                auto tmp_field_value = zfield_parser(x,y,z,t);
                auto flag_type = zflag_parser(x,y,z);
                if (flag_type != 0._rt && flag_type != 1._rt && flag_type != 2._rt) {
                    amrex::Abort("flag type for excitation must be 0, or 1, or 2!");
                } else if ( flag_type > 0._rt ) {
                    Fz(i, j, k) = Fz(i,j,k)*(flag_type-1.0_rt) + zfield_parser(x,y,z,t);
                }
            }
        );
    }

}
