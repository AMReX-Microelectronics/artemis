import numpy as np
import gdspy
from skimage.draw import polygon
import matplotlib.pyplot as plt
from util import *



# Load GDS
gdsii = gdspy.GdsLibrary()
gdsii.read_gds('crosstalk_test_v01.gds')
cell = gdsii.top_level()[0]

# Get all polygons grouped by (layer, datatype)
poly_dict = cell.get_polygons(by_spec=True)

# Extract unique layers
unique_layers = sorted(set(layer for (layer, datatype) in poly_dict.keys()))
print(f"Found {len(unique_layers)} unique layers:")
for layer in unique_layers:
    print(f"  - Layer {layer}")

# Define grid size
grid_size = (2000, 2000)
array = np.zeros(grid_size, dtype=np.uint8)

# Define which layers to include
selected_layers = {0, 1, 2, 3}  # Or choose based on what's present


all_polys = []
for (layer, datatype), polygons in poly_dict.items():
    if layer in selected_layers:
        all_polys.extend(polygons)

all_coords = np.vstack(all_polys)
x_min, y_min = all_coords.min(axis=0)
x_max, y_max = all_coords.max(axis=0)


for (layer, datatype), polygons in poly_dict.items():
    if layer not in selected_layers:
        continue
    for poly in polygons:
        x = (poly[:, 0] - x_min) / (x_max - x_min + 1e-12) * (grid_size[1] - 1)
        y = (poly[:, 1] - y_min) / (y_max - y_min + 1e-12) * (grid_size[0] - 1)
        rr, cc = polygon(y, x)
        rr, cc = rr.astype(int), cc.astype(int)
        rr = np.clip(rr, 0, grid_size[0] - 1)
        cc = np.clip(cc, 0, grid_size[1] - 1)
        array[rr, cc] = 1

array_T = array.T
array_T_clean = remove_outer_connected_ones(array_T)
plt.figure(figsize=(6, 6))
plt.imshow(array_T_clean, cmap='viridis', origin='lower')
plt.colorbar(label='Occupancy')
plt.title('Rasterized GDS on Layers {1,2,3}')
plt.xlabel('y')
plt.ylabel('x')
plt.tight_layout()
plt.show()

# x and y coordinates, note the last point is excluded
array_T_clean_new = array_T_clean[600:1180, 500:1300]
flipped_array = 1 - array_T_clean_new


# Save and plot
array_3d = np.repeat(array_T_clean_new[:, :, np.newaxis], 1, axis=2)
flipped_array_3d = np.repeat(flipped_array[:, :, np.newaxis], 1, axis=2)
array_3d = array_3d.astype(np.float64)
flipped_array_3d = flipped_array_3d.astype(np.float64)
np.save("crosstalk_test_v01.npy", array_3d)
np.save("crosstalk_test_v01_flipped.npy", flipped_array_3d)
