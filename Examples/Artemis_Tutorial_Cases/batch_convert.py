import glob
import os
import re
import numpy as np
import yt

# Directory containing your plotfile folders (e.g., plt00001000, plt00002000, etc.)
input_dir = "./diags_LLG"
output_dir = "./raw_data_LLG"

os.makedirs(output_dir, exist_ok=True)

# Find and sort all plotfile folders numerically by timestep
plt_folders = sorted(
    [d for d in glob.glob(os.path.join(input_dir, "plt*")) if os.path.isdir(d)],
    key=lambda x: int(re.search(r'plt(\d+)', x).group(1))
)

print(f"Found {len(plt_folders)} plotfiles to process.")

for plt_path in plt_folders:
    # Extract integer timestep from folder name (e.g., 'plt00001000' -> 1000)
    step_num = int(re.search(r'plt(\d+)', plt_path).group(1))

    # Load dataset and extract covering grid
    ds = yt.load(plt_path)
    ad0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)

    # Extract 3D Ey array and slice 1D profile at transverse cell (0, 0)
    E_array = ad0['Ey'].to_ndarray()
    ey_1d = E_array[0, 0, :]

    # Generate 0-based k indices
    k_indices = np.arange(len(ey_1d))

    # Format 2-column data matrix: [k_index, Ey]
    output_data = np.column_stack((k_indices, ey_1d))

    # Save output text file matching your naming scheme (e.g., 1000.txt)
    output_filename = os.path.join(output_dir, f"{step_num}.txt")
    np.savetxt(output_filename, output_data, fmt="%d  %12.8e")

print(f"Batch conversion complete! Output saved to '{output_dir}'.")
