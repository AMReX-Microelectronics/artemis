import numpy as np

def pad_to_shape(array, target_shape):
    """
    Pad a 2D NumPy array with zeros to the target shape, centering the original array.

    Parameters:
        array (np.ndarray): 2D input array to pad.
        target_shape (tuple): Desired shape (rows, cols) as (height, width).

    Returns:
        np.ndarray: Zero-padded array of shape target_shape.
    """
    if len(array.shape) != 2:
        raise ValueError("Only 2D arrays are supported.")

    if array.shape[0] > target_shape[0] or array.shape[1] > target_shape[1]:
        raise ValueError("Target shape must be greater than or equal to array shape in both dimensions.")

    pad_x = target_shape[0] - array.shape[0]
    pad_y = target_shape[1] - array.shape[1]

    pad_x_before = pad_x // 2
    pad_x_after = pad_x - pad_x_before
    pad_y_before = pad_y // 2
    pad_y_after = pad_y - pad_y_before

    padded = np.pad(
        array,
        pad_width=((pad_x_before, pad_x_after), (pad_y_before, pad_y_after)),
        mode='constant',
        constant_values=0
    )

    return padded


import numpy as np
from scipy.ndimage import binary_fill_holes, label, generate_binary_structure


def remove_outer_connected_ones(mask):
    """
    Remove all 1s that are connected to the image boundary.
    Retains only the 'inner' objects fully surrounded by 0s.

    Parameters:
        mask (np.ndarray): 2D binary array (0 and 1)

    Returns:
        np.ndarray: Cleaned binary mask
    """
    # Ensure binary
    mask = (mask > 0).astype(np.uint8)

    # Label connected components
    structure = np.ones((3, 3), dtype=np.uint8)  # 8-connectivity
    labeled, num_features = label(mask, structure=structure)

    # Find labels connected to the border
    border_labels = set()
    border_labels.update(np.unique(labeled[0, :]))  # top
    border_labels.update(np.unique(labeled[-1, :]))  # bottom
    border_labels.update(np.unique(labeled[:, 0]))  # left
    border_labels.update(np.unique(labeled[:, -1]))  # right

    # Remove border-connected labels
    cleaned_mask = np.copy(mask)
    for label_id in border_labels:
        cleaned_mask[labeled == label_id] = 0

    return cleaned_mask

import numpy as np

def generate_2d_coordinates(shape, x_bounds, y_bounds):
    """
    Generate coordinate arrays for a 2D grid.

    Parameters:
        shape (tuple): Shape of the 2D array (ny, nx)
        x_bounds (tuple): (x_lo, x_hi) bounds for x-axis (length = nx)
        y_bounds (tuple): (y_lo, y_hi) bounds for y-axis (length = ny)

    Returns:
        coords (ndarray): Array of shape (ny, nx, 2) where coords[i, j] = [x, y]
    """
    nx, ny = shape
    x_lo, x_hi = x_bounds
    y_lo, y_hi = y_bounds

    x = np.linspace(x_lo, x_hi, nx)
    y = np.linspace(y_lo, y_hi, ny)

    X, Y = np.meshgrid(x, y, indexing='xy')
    coords = np.stack((X, Y), axis=-1)

    return coords


