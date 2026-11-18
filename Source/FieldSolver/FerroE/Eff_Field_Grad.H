/*
  Derivative algorithms for calculating the gradient energy terms
*/

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDx (
   amrex::Array4<amrex::Real> const& F,
   int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
   return (F(i+1,j,k,0) - F(i,j,k,0))/(dx[0]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDx (
   amrex::Array4<amrex::Real> const& F,
   int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
   return (F(i,j,k,0) - F(i-1,j,k,0))/(dx[0]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDy (
   amrex::Array4<amrex::Real> const& F,
   int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
   return (F(i,j+1,k,0) - F(i,j,k,0))/(dx[1]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDy (
   amrex::Array4<amrex::Real> const& F,
   int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
   return (F(i,j,k,0) - F(i,j-1,k,0))/(dx[1]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDz (
   amrex::Array4<amrex::Real> const& F,
   int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
   return (F(i,j,k+1,0) - F(i,j,k,0))/(dx[2]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDz (
   amrex::Array4<amrex::Real> const& F,
   int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
   return (F(i,j,k,0) - F(i,j,k-1,0))/(dx[2]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DoubleDx (
   amrex::Array4<amrex::Real> const& F,
   int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, amrex::Array4<amrex::Real> const& fe, int const ncomp = 0) {

   if (ncomp == 0){ //x-component
      if (fe(i,j,k) == 1 && fe(i+1,j,k) == 0) {//upper end of fe

         return (0.0 - DownwardDx(F, i, j, k, dx))/dx[0];

      } else if (fe(i,j,k) == 1 && fe(i-1,j,k) == 0) { //lower end of fe

         return (UpwardDx(F, i, j, k, dx) - 0.0)/dx[0];

      } else {//bulk

         return (UpwardDx(F, i, j, k, dx) - DownwardDx(F, i, j, k, dx))/dx[0];

      }
   } else { //y and z component
      if (fe(i,j,k) == 1 && fe(i+1,j,k) == 0) {//upper end of fe

         return 0.5*(8.*F(i-1,j,k,0) - F(i-2,j,k,0) - 7.*F(i,j,k,0))/dx[0]/dx[0];

      } else if (fe(i,j,k) == 1 && fe(i-1,j,k) == 0) { //lower end of fe

         return 0.5*(8.*F(i+1,j,k,0) - F(i+2,j,k,0) - 7.*F(i,j,k,0))/dx[0]/dx[0];

      } else {//bulk

         return (UpwardDx(F, i, j, k, dx) - DownwardDx(F, i, j, k, dx))/(dx[0]);

      }

   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DoubleDy (
   amrex::Array4<amrex::Real> const& F,
   int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, amrex::Array4<amrex::Real> const& fe, int const ncomp = 0) {

   if (ncomp == 1){ //y-component
      if (fe(i,j,k) == 1 && fe(i,j+1,k) == 0) {//upper end of fe

         return (0.0 - DownwardDy(F, i, j, k, dx))/dx[1];

      } else if (fe(i,j,k) == 1 && fe(i,j-1,k) == 0) { //lower end of fe

         return (UpwardDy(F, i, j, k, dx) - 0.0)/dx[1];

      } else {//bulk

         return (UpwardDy(F, i, j, k, dx) - DownwardDy(F, i, j, k, dx))/(dx[1]);

      }
   } else { //x and z component
      if (fe(i,j,k) == 1 && fe(i,j+1,k) == 0) {//upper end of fe

         return 0.5*(8.*F(i,j-1,k,0) - F(i,j-2,k,0) - 7.*F(i,j,k,0))/dx[1]/dx[1];

      } else if (fe(i,j,k) == 1 && fe(i,j-1,k) == 0) { //lower end of fe

         return 0.5*(8.*F(i,j+1,k,0) - F(i,j+2,k,0) - 7.*F(i,j,k,0))/dx[1]/dx[1];

      } else {//bulk

         return (UpwardDy(F, i, j, k, dx) - DownwardDy(F, i, j, k, dx))/dx[1];

      }

   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DoubleDz (
   amrex::Array4<amrex::Real> const& F,
   int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, amrex::Array4<amrex::Real> const& fe, int const ncomp = 0) {

   if (ncomp == 2){ //z-component
      if (fe(i,j,k) == 1 && fe(i,j,k+1) == 0) {//upper end of fe

         return (0.0 - DownwardDz(F, i, j, k, dx))/dx[2];

      } else if (fe(i,j,k) == 1 && fe(i,j,k-1) == 0) { //lower end of fe

         return (UpwardDz(F, i, j, k, dx) - 0.0)/dx[2];

      } else {//bulk

         return (UpwardDz(F, i, j, k, dx) - DownwardDz(F, i, j, k, dx))/dx[2];

      }
   } else { //x and y component
      if (fe(i,j,k) == 1 && fe(i,j,k+1) == 0) {//upper end of fe

         return 0.5*(8.*F(i,j,k-1,0) - F(i,j,k-2,0) - 7.*F(i,j,k,0))/dx[2]/dx[2];

      } else if (fe(i,j,k) == 1 && fe(i,j,k-1) == 0) { //lower end of fe

         return 0.5*(8.*F(i,j,k+1,0) - F(i,j,k+2,0) - 7.*F(i,j,k,0))/dx[2]/dx[2];

      } else {//bulk

         return (UpwardDz(F, i, j, k, dx) - DownwardDz(F, i, j, k, dx))/dx[2];

      }

   }
}

