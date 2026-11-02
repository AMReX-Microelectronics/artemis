# Test Cases
This file contains two test cases: an air-filled waveguide and a magnetically tunable filter. The simulation parameters are defined in an input file, "/inputs_*". To simulate the air-filled waveguide using NERSC's Perlmutter, run
> sbatch perl.run
The simulation will generate a folder of diagnostic data titled "/diags". Next, to convert the diagnostic data into readable txt files, run
> python3 batch_convert.py 
The converted data will be stored in "/raw_data*". Finally, to extract the phase constant and attenuation spectrums, run 
> python3 alphaExtraction.py

This folders contains the following files:
- alphaExtraction.py : Extracts the components of the wavevector, k, from a "/raw_data*" file
- batch_convert.py : Converts diagnostic data, "/diags", into readable data, "raw_data*".
- inputs_3d_empty_X_band : Input file defining simulation parameters for air-filled waveguide case.
- inputs_3d_LLG_filter : Input file defining simulation parameters for the magnetically tunable filter.
- main3d.gnu.TPROF.MTMPI.CUDA.ex : Executable file used for air-filled waveguide; the non-LLG case.
- main3d.gnu.TPROF.MTMPI.CUDA.LLG.ex : Executable file used for the magnetically tunable filter; the LLG case.
- perl_empty.run : Run file for NERSC's Perlmutter for the non-LLG case.
- perl_llg.run : Run file for NERSC's Perlmutter for LLG case.

