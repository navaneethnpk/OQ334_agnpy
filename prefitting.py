import numpy as np
import astropy.units as u
from astropy.constants import c, G, M_sun
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import yaml
import logging
import warnings
from copy import deepcopy

# agnpy imports
from agnpy.spectra import BrokenPowerLaw
from agnpy.fit import ExternalComptonModel, load_gammapy_flux_points
from agnpy.utils.plot import load_mpl_rc, sed_y_label

# Apply agnpy's default Matplotlib style settings
load_mpl_rc()

warnings.filterwarnings("ignore")
np.seterr(all="ignore")


def setup_logging(sed_name):
    """Setup logging to file and console."""
    
    log_file = f"{sed_name}/PreFit/{sed_name}_PreFit.log"
    
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

def deep_merge(base_dict, override_dict):
    """
    Deep merge override_dict into base_dict. Values in override_dict take precedence.
    """
    result = deepcopy(base_dict)
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def create_model_config(defaults, overrides, source_info, frozen_params):
    """Create a complete model config by merging defaults with overrides."""
    cfg = {
        "source": source_info,
        "eed": deep_merge(defaults["eed"], overrides.get("eed", {})),
        "blob": deep_merge(defaults["blob"], overrides.get("blob", {})),
        "disk": deep_merge(defaults["disk"], overrides.get("disk", {})),
        "blr": deep_merge(defaults["blr"], overrides.get("blr", {})),
        "dt": deep_merge(defaults["dt"], overrides.get("dt", {})),
    }
    return cfg

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

def plot_sed(datasets, ec_model_list, output_path, labels=None):
    """Plot SED with multiple EC models on the same plot"""
    if labels is None:
        labels = [f"EC model {i+1}" for i in range(len(ec_model_list))]
    
    colors = ["k", "r", "b", "g", "orange", "purple", "cyan", "magenta"]
    linestyles = ["-", "--", ":", "-.", "-", "--", ":", "-."]
    
    fig, ax = plt.subplots(figsize=(10, 7))
 
    # Plot data
    for dataset in datasets:
        dataset.data.plot(ax=ax, label=None)
 
    # Plot all EC models
    for idx, (ec_model, label) in enumerate(zip(ec_model_list, labels)):
        ec_model.plot(
            ax=ax,
            energy_bounds=[1e-6, 1e14] * u.eV,
            energy_power=2,
            label=label,
            color=colors[idx % len(colors)],
            lw=1.6,
            linestyle=linestyles[idx % len(linestyles)],
        )
 
    ax.set_ylabel(sed_y_label)
    ax.set_xlabel(r"$E\,/\,{\rm eV}$")
    ax.set_xlim([1e-5, 1e12])
    ax.set_ylim([1e-14, 1e-8])
    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    # plt.show()

if __name__ == "__main__":
    """
    python run_prefit.py Epoch_F1 1
    """

    sed_name = sys.argv[1]
    no = sys.argv[2]

    sed_file = f"{sed_name}/{sed_name}_sed.ecsv"
    config_file = f"{sed_name}/PreFit/config.yaml"
    
    # Setup logging
    logger = setup_logging(sed_name)
    logger.info(f"{'='*70}  {sed_name}_{no}")

    logger.info("Reading the SED file ...")
    datasets = load_data(sed_file)

    # Load main config
    main_config = load_config(config_file)
    
    defaults = main_config.get("defaults", {})
    model_variations = main_config.get("models", [])
    source_info = main_config.get("source", {})
    frozen_params = main_config.get("frozen", {})
    
    logger.info(f"Found {len(model_variations)} model variation(s)")

    # Create models from each variation
    ec_models = []
    labels = []

    for i, model_var in enumerate(model_variations):
        label = model_var.get("label", f"Model {i+1}")
        
        # Extract overrides (everything except 'label')
        overrides = {k: v for k, v in model_var.items() if k != "label"}
        
        # Create complete config by merging defaults with overrides
        cfg = create_model_config(defaults, overrides, source_info, frozen_params)
        
        logger.info(f"Creating EC Model {i+1}: {label}")
        logger.info(f"  Overrides: {overrides}")
        
        logger.info(f"Loaded params [config]:\n{cfg}")
        ec_model = make_model(cfg, ec_components=["dt", "blr"], backend="gammapy")
        # logger.info(f"\n {ec_model.parameters.to_table()}")
        
        ec_models.append(ec_model)
        labels.append(label)

    logger.info("Saving combined SED plot with all models ...")
    
    plot_sed(
        datasets, 
        ec_models, 
        f"{sed_name}/PreFit/{sed_name}_PreFit_{no}.png",
        labels=labels
    )

    logger.info("Done.\n")