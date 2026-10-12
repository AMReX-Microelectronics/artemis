# ARTEMIS
ARTEMIS (Adaptive mesh Refinement Time-domain ElectrodynaMIcs Solver) is a high-performance coupled electrodynamics–micromagnetics solver for full physical modeling of signals in microelectronic circuitry. The overall strategy couples a finite-difference time-domain (FDTD) approach for Maxwell’s equations to a magnetization model described by the Landau–Lifshitz–Gilbert (LLG) equation. The algorithm is implemented in the Exascale Computing Project (ECP) software framework, AMReX, which provides effective scalability on manycore and GPU-based supercomputing architectures. Furthermore, the code leverages ongoing developments of the Exascale Application Code, WarpX, which is primarily being developed for plasma wakefield accelerator modeling. Our temporal coupling scheme provides second-order accuracy in space and time by combining the integration steps for the magnetic field and magnetization into an iterative sub-step that includes a trapezoidal temporal discretization for the magnetization. The performance of the algorithm is demonstrated by the excellent scaling results on NERSC multicore and GPU systems, with a significant (59×) speedup on the GPU using a node-by-node comparison. The utility of our code is validated by performing simulations of transmission lines, rectangle electromagnetic waveguides, magnetically tunable filters, on-chip coplanar waveguides and resonators, magnon-photon coupling circuits, and so on.

For questions, please reach out to Zhi (Jackie) Yao (jackie_zhiyao@lbl.gov) and Andy Nonaka (ajnonaka@lbl.gov).

# Installation

## Prerequisites
- C++ compiler with C++17 support (GCC, Clang, Intel, NVCC for GPU builds)
- MPI implementation (OpenMPI, MPICH) - optional but recommended
- CUDA Toolkit (for GPU builds) - optional
- OpenMP (for CPU parallel builds) - optional
- For CMake builds: CMake version 3.18 or higher
- For GNU Make builds: GNU Make

## Download Repositories

### Download AMReX Repository
```bash
git clone https://github.com/AMReX-Codes/amrex.git
```

### Download Artemis Repository
```bash
git clone https://github.com/AMReX-Microelectronics/artemis.git
```

Make sure that AMReX and Artemis are cloned in the same location in their filesystem.

## Build Options

### Option 1: Build with GNU Make

Navigate to the Exec folder of Artemis and execute:

#### Basic build
```bash
make -j 4
```

#### Build without LLG
```bash
make -j 4 USE_LLG=FALSE
```

#### Build with LLG (default)
```bash
make -j 4 USE_LLG=TRUE
```

#### GPU build with CUDA
```bash
make -j 4 USE_LLG=TRUE USE_GPU=TRUE
```

The default value of `USE_LLG` is `TRUE`.

### Option 2: Build with CMake

Create a build directory and configure:

#### Basic CPU Build
```bash
cd artemis
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j 4
```

#### MPI + OpenMP Build
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DWarpX_MPI=ON \
  -DWarpX_COMPUTE=OMP \
  -DWarpX_MAG_LLG=ON
cmake --build build -j 4
```

#### GPU Build with CUDA
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DWarpX_COMPUTE=CUDA \
  -DWarpX_MPI=ON \
  -DWarpX_MAG_LLG=ON \
  -DAMReX_CUDA_ARCH=8.0  # Adjust for your GPU architecture
cmake --build build -j 4
```

#### Build without LLG
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DWarpX_MAG_LLG=OFF
cmake --build build -j 4
```

### Common CMake Options
- `-DWarpX_MAG_LLG=ON/OFF` - Enable/disable LLG equation (default: ON)
- `-DWarpX_MPI=ON/OFF` - Enable/disable MPI (default: ON)
- `-DWarpX_COMPUTE=NOACC/OMP/CUDA/SYCL` - Set compute backend
- `-DWarpX_PRECISION=SINGLE/DOUBLE` - Set floating point precision
- `-DWarpX_EB=ON/OFF` - Enable/disable embedded boundaries
- `-DWarpX_OPENPMD=ON/OFF` - Enable/disable openPMD I/O
- `-DCMAKE_BUILD_TYPE=Debug/Release` - Set build type

### AMReX Configuration Options

**AMReX Dependency Management:**
```bash
# Use external AMReX installation
cmake -S . -B build \
  -DWarpX_amrex_internal=OFF \
  -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX

# Use local AMReX source directory
cmake -S . -B build -DWarpX_amrex_src=/path/to/amrex/source

# Use custom AMReX repository/branch
cmake -S . -B build \
  -DWarpX_amrex_repo=https://github.com/user/amrex.git \
  -DWarpX_amrex_branch=my_branch
```

# Running Artemis

Example input scripts are located in `Examples` directory.

## Running with GNU Make builds

### Simple Testcase without LLG
For an air-filled X-band rectangle waveguide:

#### MPI+OMP build
```bash
make -j 4 USE_LLG=FALSE
mpirun -n 4 ./main3d.gnu.TPROF.MTMPI.OMP.GPUCLOCK.ex Examples/Waveguide/inputs_3d_empty_X_band
```

#### MPI+CUDA build
```bash
make -j 4 USE_LLG=FALSE USE_GPU=TRUE
mpirun -n 4 ./main3d.gnu.TPROF.MTMPI.CUDA.GPUCLOCK.ex Examples/Waveguide/inputs_3d_empty_X_band
```

### Simple Testcase with LLG
For an X-band magnetically tunable filter:

#### MPI+OMP build
```bash
make -j 4 USE_LLG=TRUE
mpirun -n 8 ./main3d.gnu.TPROF.MTMPI.OMP.GPUCLOCK.ex Examples/Waveguide/inputs_3d_LLG_filter
```

#### MPI+CUDA build
```bash
make -j 4 USE_LLG=TRUE USE_GPU=TRUE
mpirun -n 8 ./main3d.gnu.TPROF.MTMPI.CUDA.GPUCLOCK.ex Examples/Waveguide/inputs_3d_LLG_filter
```

## Running with CMake builds

The CMake build produces executables in the build directory. The exact name depends on your configuration:

### Basic execution
```bash
./build/bin/warpx Examples/Waveguide/inputs_3d_empty_X_band
```

### With MPI
```bash
mpirun -n 4 ./build/bin/warpx Examples/Waveguide/inputs_3d_LLG_filter
```

### With GPU
```bash
mpirun -n 4 ./build/bin/warpx Examples/Waveguide/inputs_3d_LLG_filter
```

[Visualization](https://amrex-codes.github.io/amrex/docs_html/Visualization_Chapter.html)

### Data Analysis in Python using yt 
You can extract the data in numpy array format using yt (you can refer to this for installation and usage of [yt](https://yt-project.org/). After you have installed yt, you can do something as follows, for example, to get variable 'Ex' (x-component of electric field)
```
import yt
ds = yt.load('./plt00001000/') # for data at time step 1000
ad0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
E_array = ad0['Ex'].to_ndarray()
```
# Publications
1. Z. Yao, R. Jambunathan, Y. Zeng and A. Nonaka, A massively parallel time-domain coupled electrodynamics–micromagnetics solver. The International Journal of High Performance Computing Applications. 2022;36(2):167-181. doi:10.1177/10943420211057906
[link](https://journals.sagepub.com/doi/full/10.1177/10943420211057906)
2. S. S. Sawant, Z. Yao, R. Jambunathan and A. Nonaka, Characterization of transmission lines in microelectronic circuits Using the ARTEMIS solver, IEEE Journal on Multiscale and Multiphysics Computational Techniques, vol. 8, pp. 31-39, 2023, doi: 10.1109/JMMCT.2022.3228281
[link](https://ieeexplore.ieee.org/abstract/document/9980353)
3. R. Jambunathan, Z. Yao, R. Lombardini, A. Rodriguez, and A. Nonaka, Two-fluid physical modeling of superconducting resonators in the ARTEMIS framework, Computer Physics Communications, 291, p.108836. doi:10.1016/j.cpc.2023.108836
[link](https://www.sciencedirect.com/science/article/pii/S0010465523001819?casa_token=rWpwl8cmtUYAAAAA:rZTndzf_pqx0lo9jtTRzLLxh0tIf_AD0zHcRRJ_ciwMw-n-X2doK5RprMS4wyrO9TEw5oDZAB7Kr)
