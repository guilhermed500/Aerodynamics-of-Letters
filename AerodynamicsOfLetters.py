"""
Aerodynamic analysis of two letters in a 2D flow domain.

Requires: numpy, pillow, pyevtk
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    from pyevtk.hl import gridToVTK  
except ImportError as exc:
    raise ImportError(
        "pyevtk is required for VTK export. Install it with "
        "'python -m pip install pyevtk' in the selected Python environment."
    ) from exc
import os
import csv

# Configuration parameters

Letters = ["A", "G"]      # the two letters to compare (top, bottom)
Nx = 400                  # domain width
Font_Size = 80
Band_Margin = 40          # vertical padding around each letter's own lane
Gap_Between = 20          # gap between the two lanes (reduces wake interference)
Inlet_Velocity = 0.08     # lattice units
Reynolds = 220
Num_Steps = 4000
Output_Every = 100
Output_Dir = "vtk_output"

Band_Ny = Font_Size + Band_Margin
Ny = Band_Ny * 2 + Gap_Between

os.makedirs(Output_Dir, exist_ok=True)
Forces_Csv = os.path.join(Output_Dir, "forces.csv")

#Build letter masks for the two letters, stacked vertically

def build_letter_mask_in_band(letter, nx, ny_total, band_ny, y_offset, font_size):
    """Renders `letter` centered inside a horizontal band of height
    band_ny starting at row y_offset, within a full canvas of ny_total."""
    img = Image.new("L", (nx, ny_total), 0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), letter, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (nx - w) // 2 - bbox[0]
    y = y_offset + (band_ny - h) // 2 - bbox[1]
    draw.text((x, y), letter, fill=255, font=font)

    arr = np.array(img).T  # shape (nx, ny_total)
    arr = arr[:,::-1]
    return arr > 128

mask_top = build_letter_mask_in_band(
    Letters[0], Nx, Ny, Band_Ny, y_offset=0, font_size=Font_Size
)
mask_bottom = build_letter_mask_in_band(
    Letters[1], Nx, Ny, Band_Ny, y_offset=Band_Ny + Gap_Between, font_size=Font_Size
)
obstacle = mask_top | mask_bottom  # combined for the solver

