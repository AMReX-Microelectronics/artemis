#include "WarpX.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXUtil.H"
#include "Parser/WarpXParserWrapper.H"
#include "Parser/GpuParser.H"

using namespace amrex;

void
WarpX::ApplyExternalFieldExcitationOnGrid ()
{
    amrex::Print() << " in apply external excitation\n";
    for (int lev = 0; lev <= finest_level; ++lev) {
        if (E_excitation_grid_s == "parse_e_excitation_grid_function")
        {
            ApplyExternalFieldExcitationOnGrid(Efield_fp[lev][0].get(),
                                               Efield_fp[lev][1].get(),
                                               Efield_fp[lev][2].get(),
                                               getParser(Exfield_xt_grid_parser),
                                               getParser(Eyfield_xt_grid_parser),
                                               getParser(Ezfield_xt_grid_parser),
                                               lev );
        }
        if (B_excitation_grid_s == "parse_b_excitation_grid_function")
        {
            ApplyExternalFieldExcitationOnGrid(Bfield_fp[lev][0].get(),
                                               Bfield_fp[lev][1].get(),
                                               Bfield_fp[lev][2].get(),
                                               getParser(Bxfield_xt_grid_parser),
                                               getParser(Byfield_xt_grid_parser),
                                               getParser(Bzfield_xt_grid_parser),
                                               lev );
        }

#ifdef WARPX_MAG_LLG
        if (H_excitation_grid_s == "parse_h_excitation_grid_function")
        {
            ApplyExternalFieldExcitationOnGrid(Hfield_fp[lev][0].get(),
                                               Hfield_fp[lev][1].get(),
                                               Hfield_fp[lev][2].get(),
                                               getParser(Hxfield_xt_grid_parser),
                                               getParser(Hyfield_xt_grid_parser),
                                               getParser(Hzfield_xt_grid_parser),
                                               lev );
        }
#endif
    } // for loop over level
}

void
WarpX::ApplyExternalFieldExcitationOnGrid (
       amrex::MultiFab *mfx, amrex::MultiFab *mfy, amrex::MultiFab *mfz,
       HostDeviceParser<4> const& xfield_parser,
       HostDeviceParser<4> const& yfield_parser,
       HostDeviceParser<4> const& zfield_parser, const int lev )
{

    // Gpu vector to store Ex-Bz staggering
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
                Fx(i, j, k) += xfield_parser(x, y, z, t);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfy_stag,
                                                  problo, dx, x, y, z);
                Fy(i, j, k) += yfield_parser(x, y, z, t);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfz_stag,
                                                  problo, dx, x, y, z);
                Fz(i, j, k) += zfield_parser(x, y, z, t);
            }
        );
    }

}
