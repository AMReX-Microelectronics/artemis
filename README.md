# ARTEMIS
ARTEMIS (Adaptive mesh Refinement Time-domain ElectrodynaMIcs Solver) is a high-performance coupled electrodynamics–micromagnetics solver for full physical modeling of signals in microelectronic circuitry. The overall strategy couples a finite-difference time-domain (FDTD) approach for Maxwell’s equations to a magnetization model described by the Landau–Lifshitz–Gilbert (LLG) equation. The algorithm is implemented in the Exascale Computing Project (ECP) software framework, AMReX, which provides effective scalability on manycore and GPU-based supercomputing architectures. Furthermore, the code leverages ongoing developments of the Exascale Application Code, WarpX, which is primarily being developed for plasma wakefield accelerator modeling. Our temporal coupling scheme provides second-order accuracy in space and time by combining the integration steps for the magnetic field and magnetization into an iterative sub-step that includes a trapezoidal temporal discretization for the magnetization. The performance of the algorithm is demonstrated by the excellent scaling results on NERSC multicore and GPU systems, with a significant (59×) speedup on the GPU using a node-by-node comparison. The utility of our code is validated by performing simulations of transmission lines, rectangle electromagnetic waveguides, magnetically tunable filters, on-chip coplanar waveguides and resonators, magnon-photon coupling circuits, and so on.

For questions, please reach out to Zhi (Jackie) Yao (jackie_zhiyao@lbl.gov) and Andy Nonaka (ajnonaka@lbl.gov).

# Installation
## Download AMReX Repository
``` git clone git@github.com:AMReX-Codes/amrex.git ```
## Download Artemis Repository
``` git clone git@github.com:AMReX-Microelectronics/artemis.git ```
## Build
Make sure that the AMReX and Artemis are cloned in the same location in their filesystem. Navigate to the Exec folder of Artemis and execute
```make -j 4```. <br />
You can turn on and off the LLG equation by specifying ```USE_LLG``` during compilation. <br />
The following command compiles Artemis without LLG
```make -j 4 USE_LLG=FALSE``` <br />
The following command compiles Artemis with LLG
```make -j 4 USE_LLG=TRUE``` <br />
The default value of ```USE_LLG``` is ```TRUE```.

# Running Artemis
Example input scripts are located in `Examples` directory. 
## Simple Testcase without LLG
You can run the following to simulate an air-filled X-band rectangle waveguide:
## For MPI+OMP build
```make -j 4 USE_LLG=FALSE``` <br />
```mpirun -n 4 ./main3d.gnu.TPROF.MTMPI.OMP.GPUCLOCK.ex Examples/Waveguide/inputs_3d_empty_X_band```
## For MPI+CUDA build
```make -j 4 USE_LLG=FALSE USE_GPU=TRUE``` <br />
```mpirun -n 4 ./main3d.gnu.TPROF.MTMPI.CUDA.GPUCLOCK.ex Examples/Waveguide/inputs_3d_empty_X_band``` 
## Simple Testcase with LLG
You can run the following to simulate an X-band magnetically tunable filter:
## For MPI+OMP build
```make -j 4 USE_LLG=TRUE``` <br />
```mpirun -n 8 ./main3d.gnu.TPROF.MTMPI.OMP.GPUCLOCK.ex Examples/Waveguide/inputs_3d_LLG_filter```
## For MPI+CUDA build
```make -j 4 USE_LLG=TRUE USE_GPU=TRUE``` <br />
```mpirun -n 8 ./main3d.gnu.TPROF.MTMPI.CUDA.GPUCLOCK.ex Examples/Waveguide/inputs_3d_LLG_filter```
# Visualization and Data Analysis
Refer to the following link for several visualization tools that can be used for AMReX plotfiles. 

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
