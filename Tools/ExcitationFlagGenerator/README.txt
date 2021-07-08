EXCITATION FLAG FUNCTION GENERATOR PYTHON SCRIPT

This python script allows the user to create an excitation flag function for an ARTEMIS inputs file using coordinate data from a text file

The format of the data text file is as follows:

dx
dy
dz
x_lo y_lo z_lo x_hi y_hi z_hi

whereas the coordinates correspond to a plane
whereas dx, dy, dz correspond to 250nm
:

Excitation_Flag_Generator.py reads the data text file and determines if there are matches between lo and hi coordinates for x, y, z

If there is a match of coordinates, the script will print that there is a normal in the corresponding axis.

Every match will be sorted into a string for Ex, Ey, Ez for use in an ARTEMIS inputs file

If there is no match of coordinates, the script will state "no match" and abort

If a lo coordinate in text file is greater in value than a hi coordinate, the script will abort

The output of the python script, the strings for Ex, Ey, Ez excitation flag functions, can be copied and pasted onto an inputs file
