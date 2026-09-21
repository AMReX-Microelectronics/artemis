import glob
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import hankel, svd


def esprit(x, p, n_sources):
    """1D Total Least Squares ESPRIT (TLS-ESPRIT) algorithm."""
    N = len(x)
    H = hankel(x[: N - p + 1], x[N - p :])

    U, _, _ = svd(H, full_matrices=False)
    Us = U[:, :n_sources]

    Us1, Us2 = Us[:-1, :], Us[1:, :]

    K = np.hstack((Us1, Us2))
    _, _, Vh_k = svd(K)
    V = Vh_k.conj().T

    V12 = V[:n_sources, n_sources:]
    V22 = V[n_sources:, n_sources:]
    Psi = -V12 @ np.linalg.inv(V22)

    z = np.linalg.eigvals(Psi)
    wm = np.log(z)

    return wm


# --- Parameters & Grid Setup ---
# Physical Constants & Waveguide Setup
EP0 = 8.854e-12
MU0 = 4 * math.pi * 1e-7
CC = 3e8

# GENERAL PARAMETERS
freq = 10.5e9 # frequency
XX, YY, ZZ = 14.95e-3, 10.16e-3, 500.0e-3 # dimensions
NX, NY, NZ = 128, 4, 128 # number of cells

DX, DY, DZ = XX / NX, YY / NY, ZZ / NZ

DT = 3.11275047e-13 # time step in seconds
TP = 1.0 / freq
Tstep_end = 1000000 # maximumm time steps
Tstep_int = 100 # interval between time steps

file_pattern = "./raw_data_empty/*.txt" # data location

k0 = 1
k1 = NZ - 1
dk = 1 
NZ_record = int(math.floor(k1 / dk))

Tstep_record = math.floor((4.7 / 6.3 * Tstep_end) / Tstep_int)
Tstep_trunc = Tstep_record

Time_axis = np.arange(Tstep_int, Tstep_end + 1, Tstep_int)

# Spectral Setup
fs = 1.0 / (Tstep_int * DT)
Tstep_zeros = 0
Tstep_ext = Tstep_record + Tstep_zeros
NFFT_ext = 1 << (Tstep_ext - 1).bit_length()

f_ext = (fs / 2.0) * np.linspace(0, 1, (NFFT_ext // 2) + 1)
f_ext_GHz = f_ext / 1e9
f_ext_step = np.mean(np.diff(f_ext))

# --- Data Import ---
txt_files = sorted(
    glob.glob(file_pattern),
    key=lambda x: int(re.search(r"(\d+)\.txt$", x).group(1)) if re.search(r"(\d+)\.txt$", x) else 0,
)

Ey_obs_ret = np.zeros((NZ_record, Tstep_record))
for tcount, fname in enumerate(txt_files[:Tstep_record]):
    data = np.loadtxt(fname, skiprows=0)
    Ey_obs_ret[:, tcount] = data[:NZ_record, 1]

ZZ_axis = DZ * np.arange(k0, k1 + 1, dk)

# --- FFT Processing ---
ey_ext_rec = np.hstack(
    (
        Ey_obs_ret[:, :Tstep_trunc],
        np.zeros((NZ_record, Tstep_record - Tstep_trunc)),
        np.zeros((NZ_record, Tstep_zeros)),
    )
)

ey_extff_rec = np.fft.fft(ey_ext_rec, n=NFFT_ext, axis=1) / Tstep_ext

# --- ESPRIT Processing (Integrated into Main Script) ---
ko1_m = int(math.floor((ZZ - DZ) / (DZ * dk)))
ko2_m = int(math.floor(DZ / (DZ * dk)))

num_freqs = int(NFFT_ext // 2) + 1
beta_esprit = np.zeros(num_freqs)
alpha_esprit = np.zeros(num_freqs)
spatial_step = DZ * dk

for fpoint in range(num_freqs):
    slice_m = ey_extff_rec[ko2_m:ko1_m, fpoint]
    wm = esprit(slice_m, p=3, n_sources=1)

    beta_esprit[fpoint] = np.abs(np.imag(wm[0])) / spatial_step
    alpha_esprit[fpoint] = np.abs(np.real(wm[0])) / spatial_step

N_m = len(beta_esprit)

# --- Data Exports ---
alpha_m_db = 10 * np.log10(np.exp(2)) * (alpha_esprit / 100)

alpha_data = np.column_stack((f_ext_GHz[:N_m], alpha_m_db))
np.savetxt("alpha.txt", alpha_data, header="Frequency Forward", fmt="%.6f", comments="")

beta_data = np.column_stack((f_ext_GHz[:N_m], beta_esprit))
np.savetxt("beta.txt", beta_data, header="Frequency Forward", fmt="%.6f", comments="")

Nf_ey = int(round(10.2e9 / f_ext_step))
ey_data = np.column_stack((ZZ_axis, np.abs(ey_extff_rec[:, Nf_ey])))
np.savetxt("Ey.txt", ey_data, header="Z position |Ey|", fmt="%.6f", comments="")

# --- Target Plots ---

# 1. beta.png
plt.figure()
plt.plot(f_ext_GHz[:N_m], beta_esprit, label="Forward")
plt.xlim([8, 12])
plt.xlabel("Frequency [GHz]")
plt.ylabel(r"Phase constant $\beta$ [rad/m]")
plt.title("Dispersion relation")
plt.savefig("fig1_dispersion.png", dpi=300)
plt.close()

# 2. alpha.png
plt.figure()
plt.plot(f_ext_GHz[:N_m], alpha_esprit, label="Forward")
plt.xlim([8.5, 12])
plt.xlabel("Frequency [GHz]")
plt.ylabel(r"Attenuation rate $\alpha$ [Np/m]")
plt.title("Attenuation Spectrum")
plt.savefig("fig2_attenuation.png", dpi=300)
plt.close()

# 3. ey.png
plt.figure()
plt.plot(ZZ_axis * 1000, np.abs(ey_extff_rec[:, Nf_ey]))
plt.ylabel("|Ey| [V/m]")
plt.xlabel("z position [mm]")
plt.title(f"{f_ext_GHz[Nf_ey]:.2f} GHz")
plt.savefig("fig3_ey.png", dpi=300)
plt.close()

# --- Interpolation and Export ---
f_start, f_end = 8.5e9, 12e9
Nf_start = int(round(f_start / f_ext_step))
Nf_end = int(round(f_end / f_ext_step))

f_interp = np.linspace(f_ext_GHz[Nf_start], f_ext_GHz[Nf_end], 150)
f_source = f_ext_GHz[Nf_start : Nf_end + 1]
alpha_source = alpha_m_db[Nf_start : Nf_end + 1]

spline_interp = interp1d(f_source, alpha_source, kind="cubic", fill_value="extrapolate")
alpha_total = 4.9 * spline_interp(f_interp)

# 4. alpha_total.png
plt.figure()
plt.plot(f_interp, alpha_total, "k")
plt.xlim([8.5, 12])
plt.ylim([0, 90])
plt.xlabel("Frequency [GHz]")
plt.ylabel("Attenuation over 49mm [dB]")
plt.title('Attenuation Spectrum')
plt.savefig("fig4_atten_total.png", dpi=300)
plt.close()

alpha_tot_data = np.column_stack((f_interp, alpha_total))
np.savetxt("alpha_total.txt", alpha_tot_data, header="Frequency Attenuation", fmt="%.6f", comments="")