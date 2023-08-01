import matplotlib.pyplot as plt
import plot
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import argrelextrema

# FUNCTIONS ===================================================================

def bragg_wavelength_to_theta(wavelength, a, h, k, l, n):
    d = np.sqrt((a**2)/(h**2 + k**2 + l**2))

    theta = (1/2) * np.arcsin((n*wavelength)/(2*d))
    theta_degree = (theta/(2*np.pi))*360
                              
    return theta_degree

def bragg_theta_to_wavelength(theta_degree, a, h, k, l, n):
    d = np.sqrt((a**2)/(h**2 + k**2 + l**2))    
    omega = 2 * (theta_degree/360)*2*np.pi

    wavelength = (2*d*np.sin(omega))/n

    return wavelength

def bragg_omega_to_wavelength(omega_degree, a, h, k, l, n):
    d = np.sqrt((a**2)/(h**2 + k**2 + l**2))    
    omega = (omega_degree/360)*2*np.pi
    
    wavelength = (2*d*np.sin(omega))/n

    return wavelength

def intensity_theta_to_wavelength(intensity_theta, omega_degree, a, h, k, l, n):
    d = np.sqrt((a**2)/(h**2 + k**2 + l**2))  
    omega = (omega_degree/360)*2*np.pi
    
    intensity_wavelength = intensity_theta * ((2 * d * np.cos(omega))/n)
    
    return intensity_wavelength

def linear(x, m, t):
    return m * x + t

# VARIABLES ===================================================================

data_folder = "../Data/XAS"
pic_folder = "../Pictures/Evaluation/41"


data_files = [x for x in os.listdir(data_folder) if "STN" in x]
data_files = sorted(data_files, key=lambda x: int(x[6]))

'''
# WOLFRAM DATA ================================================================

wolfram_wavelengths = [[1.06146e-10, 1.06786e-10, 1.09857e-10, 1.24430e-10, 1.26271e-10, 1.28181e-10, 1.30164e-10, 1.42112e-10, 1.47631e-10, 1.48745e-10],
                       [0.17892e-10, 0.17892e-10, 0.17942e-10, 0.17960e-10, 0.18310e-10, 0.18327e-10, 0.18438e-10, 0.18518e-10, 0.20901e-10, 0.21383e-10]]

wolfram_labels = [[r"$L\gamma_3$", r"$L\gamma_2$", r"$L\gamma_1$", r"$L\beta_2$", r"$L\beta_3$", r"$L\beta_1$", r"$L\beta_4$", r"$L\eta$", r"$L\alpha_1$", r"$L\alpha_2$"],
                  [r"$K\beta_4^{I}$", r"$K\beta_4^{II}$", r"$K\beta_2^{I}$", r"$K\beta_2^{II}$", r"$K\beta_5^{I}$", r"$K\beta_5^{II}$", r"$K\beta_1$", r"$K\beta_3$", r"$K\alpha_1$", r"$K\alpha_2$"]]

wolfram_labels_iupac = [[r"$\mathrm{L}_1-\mathrm{N}_3$", r"$\mathrm{L}_1-\mathrm{N}_2$", r"$\mathrm{L}_2-\mathrm{N}_4$", r"$\mathrm{L}_3-\mathrm{N}_5$", r"$\mathrm{L}_1-\mathrm{M}_3$", r"$\mathrm{L}_2-\mathrm{M}_4$", r"$\mathrm{L}_1-\mathrm{M}_2$", r"$\mathrm{L}_2-\mathrm{M}_1$", r"$\mathrm{L}_3-\mathrm{M}_5$", r"$\mathrm{L}_3-\mathrm{M}_4$"],
                        [r"$\mathrm{K}-\mathrm{N}_5$", r"$\mathrm{K}-\mathrm{N}_4$", r"$\mathrm{K}-\mathrm{N}_3$", r"$\mathrm{K}-\mathrm{N}_2$", r"$\mathrm{K}-\mathrm{M}_5$", r"$\mathrm{K}-\mathrm{M}_4$", r"$\mathrm{K}-\mathrm{M}_3$", r"$\mathrm{K}-\mathrm{M}_2$", r"$\mathrm{K}-\mathrm{L}_3$", r"$\mathrm{K}-\mathrm{L}_2$"]]

wolfram_thetas1 = bragg_wavelength_to_theta(np.array(wolfram_wavelengths), 5.463e-10, 2, 2, 0, 1)
wolfram_omegas1 = 2 * wolfram_thetas1

wolfram_thetas2 = bragg_wavelength_to_theta(np.array(wolfram_wavelengths), 5.463e-10, 2, 2, 0, 2)
wolfram_omegas2 = 2 * wolfram_thetas2

wolfram_dict_L = {"Siegbahn":wolfram_labels[0],
                  "IUPAC":wolfram_labels_iupac[0],
                  "lambda[m]":wolfram_wavelengths[0],
                  "omega1[deg]":wolfram_omegas1[0],
                  "theta1[deg]":wolfram_thetas1[0],
                  "omega2[deg]":wolfram_omegas2[0],
                  "theta2[deg]":wolfram_thetas2[0]}

wolfram_dict_K = {"Siegbahn":wolfram_labels[1],
                  "IUPAC":wolfram_labels_iupac[1],
                  "lambda[m]":wolfram_wavelengths[1],
                  "omega1[deg]":wolfram_omegas1[1],
                  "theta1[deg]":wolfram_thetas1[1],
                  "omega2[deg]":wolfram_omegas2[1],
                  "theta2[deg]":wolfram_thetas2[1]}

df_wolfram_L = pd.DataFrame(wolfram_dict_L)
df_wolfram_K = pd.DataFrame(wolfram_dict_K)

df_wolfram_L.to_csv(data_folder + "/" + "wolfram_anode_L-series.csv", index=False) 
df_wolfram_K.to_csv(data_folder + "/" + "wolfram_anode_K-series.csv", index=False) 
#'''

'''
# ANGLE =======================================================================

df = pd.read_csv(data_folder + "/" + data_files[0], skiprows=22, names=["omega[deg]", "I[au]", "I_B[au]"], delim_whitespace=True)
df["theta[deg]"] = (1/2) * df["omega[deg]"]

df_wolfram_L = pd.read_csv(data_folder + "/" + "wolfram_anode_L-series.csv")

theta_1, int_1, back_1 = df["theta[deg]"][80:200], df["I[au]"][80:200], df["I_B[au]"][80:200]
theta_2, int_2, back_2 = df["theta[deg]"][260:], df["I[au]"][260:], df["I_B[au]"][260:]

int_corr_1 = int_1 - back_1
int_corr_2 = int_2 - back_2

theta_max_theo_1 = list(df_wolfram_L["theta1[deg]"])
theta_max_theo_2 = list(df_wolfram_L["theta2[deg]"])
theta_max_theo = theta_max_theo_1 + theta_max_theo_2

# achived through try and error (very annoying)
peaks_1 = [104, 105, 110, 133, 136, 142, 139, 162, 170, 172]
peaks_2 = [278, 280, 292, 347, 355, 362, 370, 420, 444, 450]

theta_max_1 = [theta_1[i] for i in peaks_1]
theta_max_2 = [theta_2[i] for i in peaks_2]
theta_max = theta_max_1 + theta_max_2

xline = np.linspace(0, 30, 1000)

plot.parameters(True, 40, (14,8), 100, colorblind = False)

popt, pcov = curve_fit(linear, theta_max, theta_max_theo)

#print(popt, np.sqrt(np.diag(pcov)))

theta_corr = popt[1] - np.sqrt(np.diag(pcov))[1]

df["omega_corr[deg]"] = 2 * linear(df["theta[deg]"], *popt)
df["theta_corr[deg]"] = linear(df["theta[deg]"], *popt)

fig, ax = plt.subplots(1, 1)

ax.plot(xline, linear(xline, *popt), linewidth=3, color = "r", label = "Linear Fit")
ax.scatter(theta_max, theta_max_theo, s = 100, label = "Data")

ax.set_xlim(0,30)
ax.set_ylim(0,30)

ax.set_xlabel(r"$\Theta_\mathrm{Data}$ [Deg]")
ax.xaxis.set_label_coords(0.5, -0.15)

ax.set_ylabel(r"$\Theta_\mathrm{Theo}$ [Deg]")
ax.yaxis.set_label_coords(-0.1, 0.5)

ax.legend(loc = "upper center", frameon = False, bbox_to_anchor = (0.5, 1.2), ncol = 2)

#plt.savefig(pic_folder + "/" + "Linear-Fit.pdf", bbox_inches = "tight")

# ROENTGENSPECTRUM ============================================================

yline_1 = np.linspace(0, 20000, 1000)
yline_2 = np.linspace(0, 700, 1000)

labels = df_wolfram_L["Siegbahn"]

plot.parameters(True, 25, (16,12), 100, colorblind = False)

fig, ax = plt.subplots(2, 1)

for theta1, theta2, label in zip(theta_max_theo_1, theta_max_theo_2, labels):
    
    ax[0].plot(np.repeat(theta1, 1000), yline_1, linewidth = 3, alpha = 0.6, label = label)
    ax[1].plot(np.repeat(theta2, 1000), yline_2, linewidth = 3, alpha = 0.6)
    
# ax[0].plot(theta_1+theta_corr, back_1, linewidth = 3, color = "black")
# ax[1].plot(theta_2+theta_corr, back_2, linewidth = 3, color = "black")    

ax[0].plot(df["theta_corr[deg]"][80:200], int_1, linewidth = 3)
ax[1].plot(df["theta_corr[deg]"][260:], int_2, linewidth = 3)

ax[0].set_xlim(7, 12)
ax[0].set_ylim(0, 20000)

ax[0].set_yticks(np.arange(0, 21000, 10000))

ax[1].set_xlim(16, 26)
ax[1].set_ylim(0, 700)

ax[1].set_xlabel(r"$\Theta_\mathrm{Data}$ [Deg]")
ax[1].xaxis.set_label_coords(0.5, -0.15)

ax[0].set_ylabel(r"Intensity [a.u.]")
ax[0].yaxis.set_label_coords(-0.08, -0.1)

ax[0].text(0.95, 0.90, r"\textbf{(a)}", transform=ax[0].transAxes)
ax[1].text(0.95, 0.90, r"\textbf{(b)}", transform=ax[1].transAxes)

ax[0].legend(loc = "upper center", frameon = False, bbox_to_anchor = (1.1, 0.5), ncol = 1, handlelength = 1)

#plt.savefig(pic_folder + "/" + "Wolfram-Spectrum.pdf", bbox_inches = "tight")
#'''

#'''
# IDENTIFICATION ==============================================================

'''
df_data = pd.read_csv(data_folder + "/" + data_files[3], skiprows=22, names=["omega[deg]", "I[au]", "I_B[au]"], delim_whitespace=True)
df_data["theta[deg]"] = (1/2) * df_data["omega[deg]"]

df_data["theta_corr[deg]"] = linear(df_data["theta[deg]"], *popt)
df_data["omega_corr[deg]"] = 2 * linear(df_data["theta[deg]"], *popt)

df_data["lambda[ang]"] = bragg_omega_to_wavelength(df_data["omega_corr[deg]"], 5.463e-10, 2, 2, 0, 1) * 1e10
df["lambda[ang]"] = bragg_omega_to_wavelength(df["omega_corr[deg]"], 5.463e-10, 2, 2, 0, 1) * 1e10

df.to_csv(data_folder + "/" + "wolfram-anode.csv", index = False)
df_data.to_csv(data_folder + "/" + "foil-4.csv", index=False)
'''

df_ref    = pd.read_csv(data_folder + "/" + "wolfram-anode.csv")
df_data_2 = pd.read_csv(data_folder + "/" + "foil-2.csv")
df_data_3 = pd.read_csv(data_folder + "/" + "foil-3.csv")
df_data_4 = pd.read_csv(data_folder + "/" + "foil-4.csv")

#intensity = intensity_theta_to_wavelength(df_data["I[au]"], df["omega[deg]"], 5.463e-10, 2, 2, 0, 1)

plot.parameters(True, 30, (14,8), 100, colorblind = False)

'''
fig, ax = plt.subplots(1, 1)

ax.plot(df_data_2["lambda[ang]"], df_data_2["I[au]"], linewidth=3, label = "Foil 2")
ax.plot(df_data_3["lambda[ang]"], 5 * df_data_3["I[au]"], linewidth=3, label = "Foil 3")
ax.plot(df_data_4["lambda[ang]"], df_data_4["I[au]"], linewidth=3, label = "Foil 4")
ax.plot(df_ref["lambda[ang]"], 3 * df_ref["I[au]"], linewidth=3, label = "Reference")

ax.set_yscale("log")

ax.set_xlim(df_ref["lambda[ang]"].min(), 1.6)
ax.set_ylim(500, 1.5e5)

ax.set_xlabel(r"$\lambda$ [\AA]")
ax.xaxis.set_label_coords(0.5, -0.1)

ax.set_ylabel(r"Intensity [a.u.]")
ax.yaxis.set_label_coords(-0.08, 0.5)

ax.legend(loc = "upper center", frameon = False, bbox_to_anchor = (0.5, 1.15), ncol = 4, handlelength = 1)

plt.savefig(pic_folder + "/" + "XRay-Spectrum.pdf", bbox_inches = "tight")
'''

x = 2.5e-5

fig, ax = plt.subplots(1, 1)

ax.scatter(df_ref["lambda[ang]"], 1/x * np.log(3 * df_ref["I[au]"]/df_data_2["I[au]"]) * 1e-5, linewidth=3, label = "Foil 2", marker = "o", s = 100)
ax.scatter(df_ref["lambda[ang]"], 1/x * np.log(3 * df_ref["I[au]"]/(5*df_data_3["I[au]"])) * 1e-5, linewidth=3, label = "Foil 3", marker = ",", s = 100)
ax.scatter(df_ref["lambda[ang]"], 1/x * np.log(3 * df_ref["I[au]"]/df_data_4["I[au]"]) * 1e-5, linewidth=3, label = "Foil 4", marker = "^", s = 100)

ax.plot(np.repeat(0.585, 1000), np.linspace(-1e4, 1.8e5, 1000) * 1e-5, linewidth = 3)
ax.plot(np.repeat(0.530, 1000), np.linspace(-1e4, 7.5e4, 1000) * 1e-5, linewidth = 3)
ax.plot(np.repeat(0.610, 1000), np.linspace(-1e4, 2.0e5, 1000) * 1e-5, linewidth = 3)

#ax.set_yscale("log")

ax.set_xlim(df_ref["lambda[ang]"].min(), 0.7)
ax.set_ylim(-0.1, 2)

ax.set_xlabel(r"$\lambda$ [\AA]")
ax.xaxis.set_label_coords(0.5, -0.1)

ax.set_ylabel(r"$\mu/\rho \times 10^5~$ [a.u.]")
ax.yaxis.set_label_coords(-0.08, 0.5)

ax.legend(loc = "upper center", frameon = False, bbox_to_anchor = (0.5, 1.15), ncol = 4, handlelength = 1)

plt.savefig(pic_folder + "/" + "Analysis-Foil.pdf", bbox_inches = "tight")

#'''