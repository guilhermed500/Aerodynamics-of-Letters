"""
Plots the aerodynamic forces over time and summary statistics for two letters.
"""
import numpy as np
import AerodynamicsOfLetters as aero
from Graphics import generate_comparison_charts
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
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

#Solver 

c = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
              [1,1],[-1,1],[-1,-1],[1,-1]])
w = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
opp = [0,3,4,1,2,7,8,5,6]

nu = aero.Inlet_Velocity * 20 / aero.Reynolds
tau = 3*nu + 0.5
omega = 1.0 / tau

Q_Dyn = 0.5 * 1.0 * aero.Inlet_Velocity**2
Chord = aero.Font_Size

def equilibrium(rho, ux, uy):
    feq = np.zeros((9, aero.Nx, aero.Ny))
    usq = ux**2 + uy**2
    for i in range(9):
        cu = c[i,0]*ux + c[i,1]*uy
        feq[i] = w[i]*rho*(1 + 3*cu + 4.5*cu**2 - 1.5*usq)
    return feq

rho = np.ones((aero.Nx, aero.Ny))
ux = np.full((aero.Nx, aero.Ny), aero.Inlet_Velocity)
uy = np.zeros((aero.Nx, aero.Ny))
ux[aero.obstacle] = 0
f = equilibrium(rho, ux, uy)

def macroscopic(f):
    rho = np.sum(f, axis=0)
    ux = np.sum(f * c[:,0,None,None], axis=0) / rho
    uy = np.sum(f * c[:,1,None,None], axis=0) / rho
    return rho, ux, uy

def stream(f):
    fnew = np.empty_like(f)
    for i in range(9):
        fnew[i] = np.roll(np.roll(f[i], c[i,0], axis=0), c[i,1], axis=1)
    return fnew

#Boundary links and forces

def find_boundary_links(letter_mask, full_obstacle):
    """
    Boundary links for ONE letter's surface only. We check against the
    full combined obstacle (so a fluid node next to the OTHER letter
    isn't mistakenly attributed here), but only keep links where the
    solid neighbor actually belongs to this letter's own mask.
    """
    links = []
    for i in range(1, 9):
        dx, dy = c[i]
        shifted_full = np.roll(np.roll(full_obstacle, -dx, axis=0), -dy, axis=1)
        shifted_this = np.roll(np.roll(letter_mask, -dx, axis=0), -dy, axis=1)
        mask = (~full_obstacle) & shifted_full & shifted_this
        xs, ys = np.where(mask)
        for x, y in zip(xs, ys):
            links.append((x, y, i))
    return links

links_top = find_boundary_links(aero.mask_top, aero.obstacle)
links_bottom = find_boundary_links(aero.mask_bottom, aero.obstacle)
print(f"Letter '{aero.Letters[0]}' (top): {len(links_top)} boundary links")
print(f"Letter '{aero.Letters[1]}' (bottom): {len(links_bottom)} boundary links")

def compute_forces(f_post_collision, links):
    Fx, Fy = 0.0, 0.0
    for (x, y, i) in links:
        fi = f_post_collision[i, x, y]
        Fx += 2 * c[i, 0] * fi
        Fy += 2 * c[i, 1] * fi
    return Fx, Fy


def export_vtk(step, ux, uy, rho, mask_top, mask_bottom):
    x = np.arange(aero.Nx+1, dtype=np.float64)
    y = np.arange(aero.Ny+1, dtype=np.float64)
    z = np.array([0.0, 1.0])

    velx = ux[:,:,None]
    vely = uy[:,:,None]
    velz = np.zeros_like(velx)
    speed = np.sqrt(ux**2 + uy**2)[:,:,None]
    pressure = (rho / 3.0)[:,:,None]
    # letter_id: 1 = top letter, 2 = bottom letter, 0 = fluid
    letter_id = np.zeros((aero.Nx, aero.Ny), dtype=np.float64)
    letter_id[mask_top] = 1.0
    letter_id[mask_bottom] = 2.0
    letter_id = letter_id[:,:,None]

    gridToVTK(
        f"{aero.Output_Dir}/flow_{step:05d}",
        x, y, z,
        cellData={
            "velocity": (velx, vely, velz),
            "speed": speed,
            "pressure": pressure,
            "letter_id": letter_id,
        }
    )

#Time loop

History={
    "step" : [],
    f"{aero.Letters[0]}_Cd": [], f"{aero.Letters[0]}_Cl": [], f"{aero.Letters[0]}_downforce": [],
    f"{aero.Letters[1]}_Cd": [], f"{aero.Letters[1]}_Cl": [], f"{aero.Letters[1]}_downforce": [],
}

with open(aero.Forces_Csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "step",
        f"{aero.Letters[0]}_Fx", f"{aero.Letters[0]}_Fy", f"{aero.Letters[0]}_Cd", f"{aero.Letters[0]}_Cl", f"{aero.Letters[0]}_downforce",
        f"{aero.Letters[1]}_Fx", f"{aero.Letters[1]}_Fy", f"{aero.Letters[1]}_Cd", f"{aero.Letters[1]}_Cl", f"{aero.Letters[1]}_downforce",
    ])

    for step in range(aero.Num_Steps):
        rho, ux, uy = macroscopic(f)

        # Inlet boundary (left) - fixed velocity, same flow hits both letters
        ux[0,:] = aero.Inlet_Velocity
        uy[0,:] = 0
        rho[0,:] = 1.0

        feq = equilibrium(rho, ux, uy)
        fout = f - omega*(f - feq)

        # Forces for each letter, from post-collision pre-streaming populations
        Fx1, Fy1 = compute_forces(fout, links_top)
        Fx2, Fy2 = compute_forces(fout, links_bottom)

        # Bounce-back on obstacle (no-slip), combined mask
        for i in range(9):
            fout[i][aero.obstacle] = f[opp[i]][aero.obstacle]

        f = stream(fout)

        Cd1, Cl1 = Fx1 / (Q_Dyn * Chord), Fy1 / (Q_Dyn * Chord)
        Cd2, Cl2 = Fx2 / (Q_Dyn * Chord), Fy2 / (Q_Dyn * Chord)
        down1 = -Fy1 if Fy1 < 0 else 0.0
        down2 = -Fy2 if Fy2 < 0 else 0.0

        writer.writerow([step, Fx1, Fy1, Cd1, Cl1, down1, Fx2, Fy2, Cd2, Cl2, down2])

        History["step"].append(step)
        History[f"{aero.Letters[0]}_Cd"].append(Cd1)
        History[f"{aero.Letters[0]}_Cl"].append(Cl1)
        History[f"{aero.Letters[0]}_downforce"].append(down1)
        History[f"{aero.Letters[1]}_Cd"].append(Cd2)
        History[f"{aero.Letters[1]}_Cl"].append(Cl2)
        History[f"{aero.Letters[1]}_downforce"].append(down2)

        if step % aero.Output_Every == 0:
            rho, ux, uy = macroscopic(f)
            export_vtk(step, ux, uy, rho, aero.mask_top, aero.mask_bottom)
            print(f"Step {step}/{aero.Num_Steps} | "
                  f"{aero.Letters[0]}: Cd={Cd1:.3f} Cl={Cl1:.3f} | "
                  f"{aero.Letters[1]}: Cd={Cd2:.3f} Cl={Cl2:.3f}")

print("Done.")
print(f"Flow fields:  {aero.Output_Dir}/flow_*.vtr  (open as a series in ParaView)")
print(f"Force history: {aero.Forces_Csv}  (compare {aero.Letters[0]} vs {aero.Letters[1]} columns)")


#Compare the two letters' lift and drag coefficients in a single plot

generate_comparison_charts(History, aero.Letters, aero.Output_Dir)