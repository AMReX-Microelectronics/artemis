/* Copyright 2021 Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_RAWEFIELDREDUCTION_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_RAWEFIELDREDUCTION_H_

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


class RawEFieldReduction : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    RawEFieldReduction(std::string rd_name);

    /**
     * This function is called at every time step, and if necessary calls the templated function
     * ComputeRawEFieldReduction(), which does the actual reduction computation.
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
    // The direction of the surface for surface integration. (e.g. X, Y, or Z)
#if (AMREX_SPACEDIM==2)
    int m_surface_normal[2]={0,0};
#else
    int m_surface_normal[3]={0,0,0};
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
    void ComputeRawEFieldReduction()
    {
        using namespace amrex::literals;

        auto & warpx = WarpX::GetInstance();
        const auto nLevel = 1;
        auto reduction_function_parser = m_parser->compile<m_nvars>();
        int integral_type = m_integral_type;
        int* surface_normal = &m_surface_normal[0];


        for (int lev = 0; lev < nLevel; ++lev) {
            const amrex::MultiFab &Ex = warpx.getEfield(lev,0);
            const amrex::MultiFab &Ey = warpx.getEfield(lev,1);
            const amrex::MultiFab &Ez = warpx.getEfield(lev,2);

            constexpr int index_Ex = 0;
            constexpr int index_Ey = 1;
            constexpr int index_Ez = 2;

            amrex::GpuArray<int, 3> Ex_IndexType{0,0,0};
            amrex::GpuArray<int, 3> Ey_IndexType{0,0,0};
            amrex::GpuArray<int, 3> Ez_IndexType{0,0,0};
            amrex::IntVect Ex_nodalType = Ex.ixType().toIntVect();
            amrex::IntVect Ey_nodalType = Ey.ixType().toIntVect();
            amrex::IntVect Ez_nodalType = Ez.ixType().toIntVect();
            for (int idim = 0; idim < 3; ++idim) {
                Ex_IndexType[idim] = Ex_nodalType[idim];
                Ey_IndexType[idim] = Ey_nodalType[idim];
                Ez_IndexType[idim] = Ez_nodalType[idim];
            }

            amrex::ReduceOps<ReduceOp> reduceEx_op;
            amrex::ReduceOps<ReduceOp> reduceEy_op;
            amrex::ReduceOps<ReduceOp> reduceEz_op;
            amrex::ReduceData<amrex::Real> reduceEx_data(reduceEx_op);
            amrex::ReduceData<amrex::Real> reduceEy_data(reduceEy_op);
            amrex::ReduceData<amrex::Real> reduceEz_data(reduceEz_op);

            using ReduceTuple = typename decltype(reduceEx_data)::Type;

            amrex::Geometry const & geom = warpx.Geom(lev);
            const amrex::RealBox& real_box = geom.ProbDomain();
            const auto dx = geom.CellSizeArray();


        // MFIter loop to interpolate fields to cell center and get maximum values
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif

            for ( amrex::MFIter mfi(Ex, false); mfi.isValid(); ++mfi)
            {
                // not including the boundary points
                const amrex::Box& tx = mfi.tilebox(Ex_nodalType);
                const amrex::Box& ty = mfi.tilebox(Ey_nodalType);
                const amrex::Box& tz = mfi.tilebox(Ez_nodalType);

                const amrex::IntVect lx = tx.smallEnd() ;
                const amrex::IntVect hx = tx.bigEnd()   - amrex::IntVect::TheNodeVector();
                const amrex::Box& tex = amrex::Box(lx,hx,Ex_nodalType);

                const amrex::IntVect ly = ty.smallEnd() ;
                const amrex::IntVect hy = ty.bigEnd()   - amrex::IntVect::TheNodeVector();
                const amrex::Box& tey = amrex::Box(ly,hy,Ey_nodalType);

                const amrex::IntVect lz = tz.smallEnd() ;
                const amrex::IntVect hz = tz.bigEnd()   - amrex::IntVect::TheNodeVector();
                const amrex::Box& tez = amrex::Box(lz,hz,Ez_nodalType);

                const auto& Ex_arr = Ex[mfi].array();
                const auto& Ey_arr = Ey[mfi].array();
                const auto& Ez_arr = Ez[mfi].array();

                reduceEx_op.eval(tex, reduceEx_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) ->ReduceTuple
                {
                    // Shift x, y, z position based on index type
                    amrex::Real fac_x = (1._rt - Ex_nodalType[0]) * dx[0] * 0.5_rt;
                    amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                    amrex::Real y = 0._rt;
                    amrex::Real fac_z = (1._rt - Ex_nodalType[1]) * dx[1] * 0.5_rt;
                    amrex::Real z = j * dx[1] + real_box.lo(1) + fac_z;
#else
                    amrex::Real fac_y = (1._rt - Ex_nodalType[1]) * dx[1] * 0.5_rt;
                    amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;
                    amrex::Real fac_z = (1._rt - Ex_nodalType[2]) * dx[2] * 0.5_rt;
                    amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;
#endif
                    return reduction_function_parser(x,y,z)*Ex_arr(i,j,k);
                });
                reduceEy_op.eval(tey, reduceEy_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) ->ReduceTuple
                {
                    // Shift x, y, z position based on index type
                    amrex::Real fac_x = (1._rt - Ey_nodalType[0]) * dx[0] * 0.5_rt;
                    amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                    amrex::Real y = 0._rt;
                    amrex::Real fac_z = (1._rt - Ey_nodalType[1]) * dx[1] * 0.5_rt;
                    amrex::Real z = j * dx[1] + real_box.lo(1) + fac_z;
#else
                    amrex::Real fac_y = (1._rt - Ey_nodalType[1]) * dx[1] * 0.5_rt;
                    amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;
                    amrex::Real fac_z = (1._rt - Ey_nodalType[2]) * dx[2] * 0.5_rt;
                    amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;
#endif
                    return reduction_function_parser(x,y,z)*Ey_arr(i,j,k);
                });
                reduceEz_op.eval(tez, reduceEz_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) ->ReduceTuple
                {
                    // Shift x, y, z position based on index type
                    amrex::Real fac_x = (1._rt - Ez_nodalType[0]) * dx[0] * 0.5_rt;
                    amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;
#if (AMREX_SPACEDIM==2)
                    amrex::Real y = 0._rt;
                    amrex::Real fac_z = (1._rt - Ez_nodalType[1]) * dx[1] * 0.5_rt;
                    amrex::Real z = j * dx[1] + real_box.lo(1) + fac_z;
#else
                    amrex::Real fac_y = (1._rt - Ez_nodalType[1]) * dx[1] * 0.5_rt;
                    amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;
                    amrex::Real fac_z = (1._rt - Ez_nodalType[2]) * dx[2] * 0.5_rt;
                    amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;
#endif
                    return reduction_function_parser(x,y,z)*Ez_arr(i,j,k);
                });
            }

            amrex::Real reducedEx_value = amrex::get<0>(reduceEx_data.value());
            amrex::Real reducedEy_value = amrex::get<0>(reduceEy_data.value());
            amrex::Real reducedEz_value = amrex::get<0>(reduceEz_data.value());

            // MPI Reduce
            // MPI reduce
            if (std::is_same<ReduceOp, amrex::ReduceOpMax>::value)
            {
                amrex::ParallelDescriptor::ReduceRealMax(reducedEx_value);
                amrex::ParallelDescriptor::ReduceRealMax(reducedEy_value);
                amrex::ParallelDescriptor::ReduceRealMax(reducedEz_value);
            }
            if (std::is_same<ReduceOp, amrex::ReduceOpMin>::value)
            {
                amrex::ParallelDescriptor::ReduceRealMin(reducedEx_value);
                amrex::ParallelDescriptor::ReduceRealMin(reducedEy_value);
                amrex::ParallelDescriptor::ReduceRealMin(reducedEz_value);
            }
            if (std::is_same<ReduceOp, amrex::ReduceOpSum>::value)
            {
                amrex::ParallelDescriptor::ReduceRealSum(reducedEx_value);
                amrex::ParallelDescriptor::ReduceRealSum(reducedEy_value);
                amrex::ParallelDescriptor::ReduceRealSum(reducedEz_value);
            // If reduction operation is a sum, multiply the value by the cell volume so that the
            // result is the integral of the function over the simulation domain.
                if (integral_type == 0) {

#if (AMREX_SPACEDIM==2)
                reducedEx_value *= dx[0]*dx[1];
                reducedEy_value *= dx[0]*dx[1];
                reducedEz_value *= dx[0]*dx[1];
#else
                reducedEx_value *= dx[0]*dx[1]*dx[2];
                reducedEy_value *= dx[0]*dx[1]*dx[2];
                reducedEz_value *= dx[0]*dx[1]*dx[2];
#endif
                } else {
#if (AMREX_SPACEDIM==2)
                amrex::Real length = surface_normal[0]*dx[1] + surface_normal[1]*dx[0];
                reducedEx_value *= length;
                reducedEy_value *= length;
                reducedEz_value *= length;
#else
                amrex::Real area = surface_normal[0]*dx[1]*dx[2] + surface_normal[1]*dx[2]*dx[0] + surface_normal[2]*dx[0]*dx[1];
                reducedEx_value *= area;
                reducedEy_value *= area;
                reducedEz_value *= area;
#endif
                }
            }
            m_data[index_Ex] = reducedEx_value;
            m_data[index_Ey] = reducedEy_value;
            m_data[index_Ez] = reducedEz_value;

        }
    }

};

#endif
