/* Copyright 2021 Revathi Jambunathan, Saurabh Sawant, Andrew Nonaka
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_RAWBFIELDREDUCTION_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_RAWBFIELDREDUCTION_H_

#include "ReducedDiags.H"
#include "WarpX.H"

#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_Reduce.H>
#include <AMReX_Tuple.H>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

class RawBFieldReduction : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    RawBFieldReduction(std::string rd_name);

    /**
     * This function is called at every time step, and if necessary calls the templated function
     * ComputeRawBFieldReduction(), which does the actual reduction computation.
     *
     * @param[in] step the timestep
     */
    virtual void ComputeDiags(int step) override final;

private:
    /// Parser to read expression to be reduced from the input file.
    /// 3 elements are x, y, z
    static constexpr int m_nvars = 3;
    std::unique_ptr<amrex::Parser> m_parser;

    // Type of reduction (e.g. Maximum, Minimum or Sum)
    int m_reduction_type;
    // Type of integration (e.g. volume or surface)
    int m_integral_type;
#if (AMREX_SPACEDIM==2)
    // The direction of the surface for surface integration. (e.g. X or Z)
    int m_surface_normal[2]={0,0};
    // Multipliers for Ex and Ez surface integrals; default 1.0
    std::vector<amrex::Real> m_scaling_factor={1.0,1.0};
#else
    // The direction of the surface for surface integration. (e.g. X, Y, or Z)
    int m_surface_normal[3]={0,0,0};
    // Multipliers for Ex, Ey, and Ez surface integrals; default 1.0
    std::vector<amrex::Real> m_scaling_factor={1.0,1.0,1.0};
#endif
public:

    /**
     * This function does the actual reduction computation.
     * The reduction operation is performed on the raw fields using amrex::ReduceOps.
     *
     * \tparam ReduceOp the type of reduction that is performed. This is typically
     *         amrex::ReduceOpMax, amrex::ReduceOpMin or amrex::ReduceOpSum.
     */
    template<typename ReduceOp>
    void ComputeRawBFieldReduction()
    {
        using namespace amrex::literals;

        auto & warpx = WarpX::GetInstance();
        const auto nLevel = 1;
        auto reduction_function_parser = m_parser->compile<m_nvars>();
        int integral_type = m_integral_type;
        int* surface_normal = &m_surface_normal[0];

        for (int lev = 0; lev < nLevel; ++lev) {
            const amrex::MultiFab &Bx = warpx.getBfield(lev,0);
            const amrex::MultiFab &By = warpx.getBfield(lev,1);
            const amrex::MultiFab &Bz = warpx.getBfield(lev,2);

            constexpr int index_Bx = 0;
            constexpr int index_By = 1;
            constexpr int index_Bz = 2;
            amrex::IntVect Bx_nodalType = Bx.ixType().toIntVect();
            amrex::IntVect By_nodalType = By.ixType().toIntVect();
            amrex::IntVect Bz_nodalType = Bz.ixType().toIntVect();

            amrex::ReduceOps<ReduceOp> reduceBx_op;
            amrex::ReduceOps<ReduceOp> reduceBy_op;
            amrex::ReduceOps<ReduceOp> reduceBz_op;
            amrex::ReduceData<amrex::Real> reduceBx_data(reduceBx_op);
            amrex::ReduceData<amrex::Real> reduceBy_data(reduceBy_op);
            amrex::ReduceData<amrex::Real> reduceBz_data(reduceBz_op);

            using ReduceTuple = typename decltype(reduceBx_data)::Type;

            amrex::Geometry const & geom = warpx.Geom(lev);
            const amrex::RealBox& real_box = geom.ProbDomain();
            const auto dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for ( amrex::MFIter mfi(Bx, false); mfi.isValid(); ++mfi)
            {
                const amrex::Box& tx = mfi.tilebox(Bx_nodalType);
                const amrex::Box& ty = mfi.tilebox(By_nodalType);
                const amrex::Box& tz = mfi.tilebox(Bz_nodalType);

                const amrex::IntVect lx = tx.smallEnd();
                const amrex::IntVect hx = tx.bigEnd();

                const amrex::IntVect ly = ty.smallEnd();
                const amrex::IntVect hy = ty.bigEnd();

                const amrex::IntVect lz = tz.smallEnd();
                const amrex::IntVect hz = tz.bigEnd();

                const auto& Bx_arr = Bx[mfi].array();
                const auto& By_arr = By[mfi].array();
                const auto& Bz_arr = Bz[mfi].array();

                reduceBx_op.eval(tx, reduceBx_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) ->ReduceTuple
                {
                    // Shift x, y, z position based on index type
                    amrex::Real fac_x = (1._rt - Bx_nodalType[0]) * dx[0] * 0.5_rt;
                    amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                    amrex::Real y = 0._rt;
                    amrex::Real fac_z = (1._rt - Bx_nodalType[1]) * dx[1] * 0.5_rt;
                    amrex::Real z = j * dx[1] + real_box.lo(1) + fac_z;
#else
                    amrex::Real fac_y = (1._rt - Bx_nodalType[1]) * dx[1] * 0.5_rt;
                    amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;
                    amrex::Real fac_z = (1._rt - Bx_nodalType[2]) * dx[2] * 0.5_rt;
                    amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;
#endif
                    // We want to weight interior faces by 1
                    // and faces that are on the faces of the box by 0.5
                    // Faces for Bx computation can lie on xmin and xmax box faces.
                    amrex::Real weight = 1.0;
                    if (i == lx[0] || i == hx[0]) {
                        weight *= 0.5;
                    }
                    return weight*reduction_function_parser(x,y,z)*Bx_arr(i,j,k);
                });
                reduceBy_op.eval(ty, reduceBy_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) ->ReduceTuple
                {
                    // Shift x, y, z position based on index type
                    amrex::Real fac_x = (1._rt - By_nodalType[0]) * dx[0] * 0.5_rt;
                    amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                    amrex::Real y = 0._rt;
                    amrex::Real fac_z = (1._rt - By_nodalType[1]) * dx[1] * 0.5_rt;
                    amrex::Real z = j * dx[1] + real_box.lo(1) + fac_z;
#else
                    amrex::Real fac_y = (1._rt - By_nodalType[1]) * dx[1] * 0.5_rt;
                    amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;
                    amrex::Real fac_z = (1._rt - By_nodalType[2]) * dx[2] * 0.5_rt;
                    amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;
#endif
                    // We want to weight interior faces by 1
                    // and faces that are on the faces of the box by 0.5
                    // Faces for By computation can lie on ymin and ymax box faces.
                    amrex::Real weight = 1.0;
                    if (j == ly[1] || j == hy[1]) {
                        weight *= 0.5;
                    }
                    return weight*reduction_function_parser(x,y,z)*By_arr(i,j,k);
                });
                reduceBz_op.eval(tz, reduceBz_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) ->ReduceTuple
                {
                    // Shift x, y, z position based on index type
                    amrex::Real fac_x = (1._rt - Bz_nodalType[0]) * dx[0] * 0.5_rt;
                    amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                    amrex::Real y = 0._rt;
                    amrex::Real fac_z = (1._rt - Bz_nodalType[1]) * dx[1] * 0.5_rt;
                    amrex::Real z = j * dx[1] + real_box.lo(1) + fac_z;
#else
                    amrex::Real fac_y = (1._rt - Bz_nodalType[1]) * dx[1] * 0.5_rt;
                    amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;
                    amrex::Real fac_z = (1._rt - Bz_nodalType[2]) * dx[2] * 0.5_rt;
                    amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;
#endif
                    // We want to weight interior faces by 1
                    // and faces that are on the faces of the box by 0.5
                    // Faces for Bz computation can lie on zmin and zmax box faces.
                    amrex::Real weight = 1.0;
                    if (k == lz[2] || k == hz[2]) {
                        weight *= 0.5;
                    }
                    return weight*reduction_function_parser(x,y,z)*Bz_arr(i,j,k);
                });
            }

            amrex::Real reducedBx_value = amrex::get<0>(reduceBx_data.value());
            amrex::Real reducedBy_value = amrex::get<0>(reduceBy_data.value());
            amrex::Real reducedBz_value = amrex::get<0>(reduceBz_data.value());

            // MPI reduce
            if (std::is_same<ReduceOp, amrex::ReduceOpMax>::value)
            {
                amrex::ParallelDescriptor::ReduceRealMax(reducedBx_value);
                amrex::ParallelDescriptor::ReduceRealMax(reducedBy_value);
                amrex::ParallelDescriptor::ReduceRealMax(reducedBz_value);
            }
            if (std::is_same<ReduceOp, amrex::ReduceOpMin>::value)
            {
                amrex::ParallelDescriptor::ReduceRealMin(reducedBx_value);
                amrex::ParallelDescriptor::ReduceRealMin(reducedBy_value);
                amrex::ParallelDescriptor::ReduceRealMin(reducedBz_value);
            }
            if (std::is_same<ReduceOp, amrex::ReduceOpSum>::value)
            {
                amrex::ParallelDescriptor::ReduceRealSum(reducedBx_value);
                amrex::ParallelDescriptor::ReduceRealSum(reducedBy_value);
                amrex::ParallelDescriptor::ReduceRealSum(reducedBz_value);

                // If reduction operation is an integral, multiply the value by the cell volume
                // If reduction operation is a surface, multiply the value by the cell face area
                if (integral_type == 0) {

#if (AMREX_SPACEDIM==2)
                amrex::Real area = dx[0]*dx[1];
                reducedBx_value *= area;
                reducedBy_value *= area;
                reducedBz_value *= area;
#else
                amrex::Real volume = dx[0]*dx[1]*dx[2];
                reducedBx_value *= volume;
                reducedBy_value *= volume;
                reducedBz_value *= volume;
#endif
                } else {
#if (AMREX_SPACEDIM==2)
                amrex::Real length = surface_normal[0]*dx[1] + surface_normal[1]*dx[0];
                reducedBx_value *= m_scaling_factor[0]*length;
                reducedBy_value *= length;
                reducedBz_value *= m_scaling_factor[1]*length;
#else
                amrex::Real area = surface_normal[0]*dx[1]*dx[2] + surface_normal[1]*dx[2]*dx[0] + surface_normal[2]*dx[0]*dx[1];
                reducedBx_value *= m_scaling_factor[0]*area;
                reducedBy_value *= m_scaling_factor[1]*area;
                reducedBz_value *= m_scaling_factor[2]*area;
#endif
                }
            }
            m_data[index_Bx] = reducedBx_value;
            m_data[index_By] = reducedBy_value;
            m_data[index_Bz] = reducedBz_value;

        }
    }

};

#endif
