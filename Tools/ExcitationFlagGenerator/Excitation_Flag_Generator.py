####################################################
####################################################
############ Excitation_Flag_Generator ###########
############## For ARTEMIS Simulations #############
####################################################
####################################################


##### function to create string for input file #####

import sys
def listToString(finalstring):
    str1 = ""
    for ele in finalstring:
        str1 += ele
    return str1

##### open text file and read data #####
print()
with open('exampledata_Excitation_Flag_Generator.txt') as myfile:

##### formatting text file for extracting data #####

    data = [(line.strip()).split() for line in myfile]

##### derivative data #####

    d_lists = data[0:3]
    d_str = [val for sublist in d_lists for val in sublist]
    d_float = [float(i) for i in d_str]
    dx = d_float[0]
    dy = d_float[1]
    dz = d_float[2]
    dx1 = dx/2
    dy1 = dy/2
    dz1 = dz/2

##### coordinate data #####

    coordinate_data = data[3:]
    ExString = []
    EyString = []
    EzString = []

##### testing coordinates to find a normal #####

    for coordinate in coordinate_data:
        x_lo = coordinate[0]
        x_hi = coordinate[3]
        y_lo = coordinate[1]
        y_hi = coordinate[4]
        z_lo = coordinate[2]
        z_hi = coordinate[5]
        datastring = f"(x >= {x_lo} - ({dx1})) * (x <= {x_hi} + ({dx1})) * (y >= {y_lo} - ({dy1})) * (y <= {y_hi} + ({dy1})) * (z >= {z_lo} - ({dz1})) * (z <= {z_hi} + ({dz1})) + "
        if x_lo == x_hi and y_lo < y_hi and z_lo < z_hi:
            print("normal in x")
            EyString.append(datastring)
            EzString.append(datastring)
        if y_lo == y_hi and x_lo < x_hi and z_lo < z_hi:
            print("normal in y")
            ExString.append(datastring)
            EzString.append(datastring)
        if z_lo == z_hi and x_lo < x_hi and y_lo < y_hi:
            print("normal in z")
            ExString.append(datastring)
            EyString.append(datastring)
        if x_lo != x_hi and y_lo != y_hi and z_lo != z_hi:
            print("Error: no match")
            sys.exit()
        if x_lo > x_hi or y_lo > y_hi or z_lo > z_hi:
            print("Error: low coordinate is greater than high coordinate")
            sys.exit()
##### Formats datastring to remove "+" at the end of the string #####

    x = listToString(ExString)[:-3]
    y = listToString(EyString)[:-3]
    z = listToString(EzString)[:-3]

##### Prints datastring for normal in x, y, z #####
    print()
    print("warpx.Ex_excitation_flag_function(x,y,z) = " + '"' + x + '"')
    print()
    print("warpx.Ey_excitation_flag_function(x,y,z) = " + '"' + y + '"')
    print()
    print("warpx.Ez_excitation_flag_function(x,y,z) = " + '"' + z + '"')
