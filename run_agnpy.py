"""
To run the agnpy in the server.
"""
import numpy as np
import astropy.units as u
from astropy.constants import c, G, M_sun
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import yaml
import logging
import warnings

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# agnpy imports
from agnpy.spectra import BrokenPowerLaw, LogParabola
from agnpy.fit import ExternalComptonModel, load_gammapy_flux_points
from agnpy.utils.plot import load_mpl_rc, sed_y_label

# Apply agnpy's default Matplotlib style settings
load_mpl_rc()

# Gammapy imports
from gammapy.modeling.models import SkyModel
from gammapy.modeling import Fit


def setup_logging(sed_name):
    """Setup logging to file and console."""
    
    log_file = f"{sed_name}/{sed_name}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(path):
    """Load parameter yaml file"""
    with open(path) as f:
        return yaml.safe_load(f)

def make_eed(cfg):
    """Make electron energy distribution from config."""
    n_e = cfg["eed"]

    eed = BrokenPowerLaw(
            k        = float(n_e["k"]) * u.Unit("cm-3"),
            p1       = float(n_e["p1"]),
            p2       = float(n_e["p2"]),
            gamma_b  = float(n_e["gamma_b"]),
            gamma_min= float(n_e["gamma_min"]),
            gamma_max= float(n_e["gamma_max"]),
        )

    return eed

def make_blob(cfg):
    """Read blob parameters from config."""
    blob = cfg["blob"]

    Gamma   = float(blob["Gamma"])
    delta_D = float(blob["delta_D"])

    Beta    = np.sqrt(1 - 1 / Gamma**2)
    mu_s    = (1 - 1 / (Gamma * delta_D)) / Beta

    t_var = float(blob["t_var"]) * u.h 
    r = float(blob["r"]) * u.cm

    return {"Gamma": Gamma, 
        "delta_D": delta_D, 
        "Beta": Beta, 
        "mu_s": mu_s,
        "B": float(blob["B"]), 
        "t_var": t_var, 
        "r": r
    }

def make_disk(cfg):
    """Read accretion-disk parameters from config."""
    ad = cfg["disk"]

    L_disk = float(ad["L_disk"]) * u.Unit("erg s-1")
    M_BH = float(ad["M_BH"]) * M_sun
    eta = float(ad["eta"])

    m_dot = (L_disk / (eta * c ** 2)).to("g s-1") 
    R_g = ((G * M_BH) / c ** 2).to("cm")
    R_in = float(ad["R_in"]) * R_g 
    R_out = float(ad["R_out"]) * R_g 

    return {
        "L_disk": L_disk,
        "M_BH":   M_BH,
        "m_dot":  m_dot,
        "R_in":   R_in,
        "R_out":  R_out,
    }

def make_blr(L_disk, cfg):
    """Read Broad-Line Region parameters from config."""
    blr = cfg["blr"]

    xi_line = float(blr["xi_line"])
    R_line = 1e17 * np.sqrt(L_disk.to_value("erg s-1") / 1e45) * u.cm
    
    return {
        "xi_line": xi_line,
        "R_line": R_line,
    }
 
def make_dt(L_disk, cfg):
    """Read Dusty Torus parameters from config."""
    dt = cfg["dt"]

    xi_dt = float(dt["xi_dt"])
    T_dt = float(dt["T_dt"]) * u.K
    R_dt = 2.5 * 1e18 * np.sqrt(L_disk.to_value("erg s-1") / 1e45) * u.cm

    return {
        "xi_dt": xi_dt,
        "T_dt": T_dt,
        "R_dt": R_dt,
    }

def make_model(cfg, ec_components, backend="gammapy"):
    """Make the ExternalComptonModel and set all parameter values.
    ec_components : list of EC targets, e.g. ["dt", "blr"], ["dt"], ["blr"]
    backend       : fitting backend (default: "gammapy")
    """
    n_e  = make_eed(cfg)
    blob = make_blob(cfg)
    disk = make_disk(cfg)

    L_disk = disk["L_disk"]

    blr = make_blr(L_disk, cfg) if "blr" in ec_components else None
    dt  = make_dt(L_disk, cfg)  if "dt"  in ec_components else None

    ec_model = ExternalComptonModel(n_e, ec_components, ssa=True, backend=backend)

    ec_model.parameters["z"].value            = cfg["source"]["z"]
    ec_model.parameters["delta_D"].value      = blob["delta_D"]
    ec_model.parameters["log10_B"].value      = np.log10(blob["B"])
    ec_model.parameters["mu_s"].value         = blob["mu_s"]
    ec_model.parameters["t_var"].value        = blob["t_var"].to_value("s")
    ec_model.parameters["log10_r"].value      = np.log10(blob["r"].to_value("cm"))

    ec_model.parameters["log10_L_disk"].value = np.log10(L_disk.to_value("erg s-1"))
    ec_model.parameters["M_BH"].value         = disk["M_BH"].to_value("g")
    ec_model.parameters["m_dot"].value        = disk["m_dot"].to_value("g s-1")
    ec_model.parameters["R_in"].value         = disk["R_in"].to_value("cm")
    ec_model.parameters["R_out"].value        = disk["R_out"].to_value("cm")

    if blr is not None:
        ec_model.parameters["xi_line"].value  = blr["xi_line"]
        ec_model.parameters["R_line"].value   = blr["R_line"].to_value("cm")

    if dt is not None:
        ec_model.parameters["xi_dt"].value    = dt["xi_dt"]
        ec_model.parameters["T_dt"].value     = dt["T_dt"].to_value("K")
        ec_model.parameters["R_dt"].value     = dt["R_dt"].to_value("cm")

    frozen_dict = cfg.get("frozen") or {}
    for param_name, frozen in frozen_dict.items():
        ec_model.parameters[f"{param_name}"].frozen = frozen

    # bounds_dict = cfg.get("bounds") or {}
    # for param_name, bounds in bounds_dict.items():
    #     param = ec_model.parameters[param_name]
    #     if param_name.startswith("log10_"):
    #         param.min = np.log(float(bounds["min"]))
    #         param.max = np.log(float(bounds["max"]))
    #     else:
    #         param.min = float(bounds["min"])
    #         param.max = float(bounds["max"])

    return ec_model

def load_data(sed_file):
    """Load flux-point datasets with energy cuts and systematic uncertainties."""
    systematics = {
        "MAGIC":     0.30,
        "Fermi-LAT": 0.10,
        "XRT":       0.10,
        "UVOT":      0.05,
    }

    E_min = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())
    E_max = 200 * u.TeV

    return load_gammapy_flux_points(
        sed_file, 
        E_min, 
        E_max, 
        systematics
    )

def plot_sed(datasets, ec_model, output_path):
    """Plot SED"""
    fig, ax = plt.subplots(figsize=(8, 6))
 
    for dataset in datasets:
        dataset.data.plot(ax=ax, label=dataset.name)
 
    ec_model.plot(
        ax=ax,
        energy_bounds=[1e-6, 1e14] * u.eV,
        energy_power=2,
        label="EC model",
        color="k",
        lw=1.6,
    )
 
    ax.set_ylabel(sed_y_label)
    ax.set_xlabel(r"$E\,/\,{\rm eV}$")
    ax.set_xlim([1e-5, 1e12])
    ax.set_ylim([1e-13, 1e-8])
    ax.legend(ncol=4, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    # plt.show()

def save_results(results, ec_model, sed_name, opath):
    """Save fit results summary and parameter table."""
    txt_path = f"{opath}/{sed_name}_fit_results.txt"
    csv_path = f"{opath}/{sed_name}_fit_params.csv"
 
    with open(txt_path, "w") as f:
        f.write(str(results))

    params_table = ec_model.parameters.to_table()
    # Add physical values for log-scale parameters
    physical_values = []
    for row in params_table:
        name = row['name']
        if name.startswith('log10_'):
            physical_values.append(10 ** row['value'])
        else:
            physical_values.append(row['value'])
    params_table['physical_value'] = physical_values
    params_table.write(csv_path, format="csv", overwrite=True)

def fit_sed(sed_file, sed_name, cfg, opath):
    """Main: build model, load data, fit, save outputs.""" 
    logger = setup_logging(sed_name)
    logger.info(f"{'='*60}  {opath}")

    logger.info("Making the EC Model ...")
    ec_model = make_model(cfg, ec_components=["dt", "blr"], backend="gammapy")
    logger.info(f"\n {ec_model.parameters.to_table()}")
    
    logger.info("Reading the SED file ...")
    datasets = load_data(sed_file)
    sky_model = SkyModel(spectral_model=ec_model, name=cfg["source"]["name"])
    datasets.models = [sky_model]
 
    logger.info("Saving pre-fit SED ...")
    plot_sed(datasets, ec_model, f"{opath}/{sed_name}_prefit.png")

    logger.info("Running fit ...")
    fitter = Fit()
    results = fitter.run(datasets)
    logger.info(f"\n {results}")
 
    logger.info("Saving fit outputs ...")
    save_results(results, ec_model, sed_name, opath)
    plot_sed(datasets, ec_model, f"{opath}/{sed_name}_bestfit.png")
 
    logger.info("Done.\n")

if __name__ == "__main__":
    """
    Example: python run_agnpy.py Epoch_F1/itr1
    """
    fit_path = sys.argv[1]
    
    sed_name = fit_path.split('/')[0]
    sed_file = f"{sed_name}/{sed_name}_sed.ecsv"
    cfg      = load_config(f"{fit_path}/config.yaml")

    fit_sed(sed_file=sed_file, sed_name=sed_name, cfg=cfg, opath=fit_path)