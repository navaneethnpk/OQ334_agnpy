"""
ref: https://github.com/Adbhavna1369/PKS1510_modelling/blob/main/plotting.py
"""
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.constants import c, G, M_sun
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
import sys
import yaml
import logging
import warnings

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# agnpy imports
from agnpy.spectra import BrokenPowerLaw, LogParabola
from agnpy.fit import ExternalComptonModel, load_gammapy_flux_points
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.targets import SSDisk, RingDustTorus, SphericalShellBLR
from agnpy.utils.plot import load_mpl_rc, sed_y_label, sed_x_label, plot_sed

# Apply agnpy's default Matplotlib style settings
load_mpl_rc()

# Gammapy imports
from gammapy.modeling.models import SkyModel


def load_config(path):
    """Load parameter yaml file."""
    with open(path) as f:
        return yaml.safe_load(f)

def load_params(path):
    """Load fitted parameters from CSV file."""
    # params_df = pd.read_csv(path)
    # params = {}
    # for _, row in params_df.iterrows():
    #     params[row['name']] = row['physical_value']
    # return params
    params_table = Table.read(path, format='ascii.csv')
    params = {}
    for row in params_table:
        params[row['name']] = row['physical_value']
    return params

def load_data(path):
    """Load SED data from ECSV file."""
    data = Table.read(path)

    e = data["e_ref"]
    y = data["e2dnde"]
    y_errp = data["e2dnde_errp"]
    y_errn = data["e2dnde_errn"]
    instrument = data["instrument"]

    nu = e.to(u.Hz, equivalencies=u.spectral())
    
    systematics = {
        "UVOT": 0.05,
        "XRT":  0.10,
        "LAT":  0.10,
    }

    y_errp_total = []
    y_errn_total = []
    for inst_i, y_i, errp_i, errn_i in zip(instrument, y, y_errp, y_errn):
        sys_frac = systematics.get(inst_i, 0.0)
        sys_err = sys_frac * y_i
        errp_tot = np.sqrt(errp_i**2 + sys_err**2)
        errn_tot = np.sqrt(errn_i**2 + sys_err**2)
        y_errp_total.append(errp_tot)
        y_errn_total.append(errn_tot)

    new_data = Table()
    new_data["instrument"] = instrument
    new_data["nu"] = nu
    new_data["e2dnde"] = y
    new_data["errn_total"] = u.Quantity(y_errn_total)
    new_data["errp_total"] = u.Quantity(y_errp_total)

    return new_data


if __name__ == "__main__":
    """
    example: python plot_agnpy.py Epoch_F1/itr2
    """

    opath    = sys.argv[1]
    sed_name = opath.split('/')[0]

    params   = load_params(f"{opath}/{sed_name}_fit_params.csv")
    cfg      = load_config(f"{opath}/config.yaml")
    
    sed      = load_data(f"{sed_name}/{sed_name}_sed.ecsv")

    # ---------------------------------------------------------------
    z       = float(params["z"])
    Gamma   = cfg["blob"]["Gamma"]
    delta_D = float(params["delta_D"])
    R_b     = (c * float(params["t_var"]) * u.s * delta_D / (1 + float(params["z"]))).to("cm")
    B       = float(params["log10_B"]) * u.G
    r       = float(params["log10_r"]) * u.cm

    n_e = BrokenPowerLaw(
        k        = float(params["log10_k"]) * u.Unit("cm-3"),
        p1       = float(params["p1"]),
        p2       = float(params["p2"]),
        gamma_b  = float(params["log10_gamma_b"]),
        gamma_min= float(params["log10_gamma_min"]),
        gamma_max= float(params["log10_gamma_max"]),
    )

    # blob definition
    blob = Blob(R_b=R_b, z=z, delta_D=delta_D, Gamma=Gamma, B=B, n_e=n_e)

    # Disk
    L_disk = float(params["log10_L_disk"]) * u.Unit("erg s-1")
    M_BH   = float(params["M_BH"]) * u.Unit("g")
    m_dot  = float(params["m_dot"]) * u.Unit("g s-1")
    eta    = (L_disk / (m_dot * c ** 2)).to_value("")
    R_in   = float(params["R_in"]) * u.cm
    R_out  = float(params["R_out"]) * u.cm
    disk   = SSDisk(M_BH, L_disk, eta, R_in, R_out)
    # DT
    xi_dt = float(params["xi_dt"])
    T_dt  = float(params["T_dt"]) * u.K
    R_dt  = float(params["R_dt"]) * u.cm
    dt    = RingDustTorus(L_disk, xi_dt, T_dt, R_dt=R_dt)
    # BLR
    xi_line     = float(params["xi_line"])
    R_line      = float(params["R_line"]) * u.cm
    lambda_line = float(params["lambda_line"]) * u.AA
    blr         = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)


    syn = Synchrotron(blob, ssa=True)
    ssc = SynchrotronSelfCompton(blob, syn)
    ec_dt = ExternalCompton(blob, dt, r)
    ec_blr = ExternalCompton(blob, blr, r)

    nu = np.logspace(9, 27, 200) * u.Hz

    syn_sed = syn.sed_flux(nu)
    ssc_sed = ssc.sed_flux(nu)
    ec_dt_sed = ec_dt.sed_flux(nu)
    ec_blr_sed = ec_blr.sed_flux(nu)
    disk_bb_sed = disk.sed_flux(nu, z)
    dt_bb_sed = dt.sed_flux(nu, z)

    total_sed = syn_sed + ssc_sed + ec_dt_sed + disk_bb_sed + ec_blr_sed

    fig, ax = plt.subplots()
    ax.loglog()

    ax.loglog(nu / (1 + z), total_sed,  ls="-",  lw=2.1, color="black", label="agnpy, total")
    ax.loglog(nu / (1 + z), syn_sed, ls="--", lw=1.3, color="goldenrod", label="synchrotron",)
    ax.loglog(nu / (1 + z), ssc_sed, ls="--",  lw=1.3,  color="dodgerblue",  label="SSC")
    ax.loglog(nu / (1 + z), ec_dt_sed, ls="--", lw=1.3, color="lightseagreen", label="EC on DT",)
    ax.loglog(nu / (1 + z), disk_bb_sed, ls="-.", lw=1.3, color="dimgray", label="disk blackbody",)
    ax.loglog(nu / (1 + z), dt_bb_sed, ls=":", lw=1.3, color="dimgray", label="DT blackbody",)
    ax.loglog(nu / (1 + z), ec_blr_sed, ls="--", lw=1.3, color="violet", label="EC on BLR",)

    colors = {"UVOT": "orange", "XRT": "blue", "LAT": "red", "MAGIC": "green"}
    for inst in np.unique(sed["instrument"]):
        mask = sed["instrument"] == inst

        ax.errorbar(
            sed["nu"][mask].value,
            sed["e2dnde"][mask].value,
            yerr=[sed["errn_total"][mask], sed["errp_total"][mask]],
            fmt="o",
            label=inst,
            color=colors.get(inst, None),
            markersize=4,
        )

    ax.set_xlabel(sed_x_label)
    ax.set_ylabel(sed_y_label)
    ax.set_xlim([1e9, 1e29])
    ax.set_ylim([10 ** (-14), 10 ** (-7)])
    ax.legend(loc="upper center", fontsize=10, ncol=2,)
    fig.savefig(f"{opath}/SED_{sed_name}.png")
    # plt.show()
    plt.close(fig)