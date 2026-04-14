### agnpy SED Modeling - OQ 334

#### `run_agnpy.py`:

**Structure**
```
.
├── plot_agnpy.py                       # SED plotting script
├── prefitting.py                       # Model pre-fitting script
├── run_agnpy.py                        # Main SED fitting script
├── Epoch_F1/                           # Epoch directory
│   ├── Epoch_F1_sed.ecsv               # SED data
│   ├── Epoch_F1.log                    # log of SED fitting
│   ├── PreFit/                         # Pre-fitting outputs and config
│   └── run1/                           # Fitting iteration dir.
│       ├── config.yaml                 # Model configuration for fitting
│       ├── Epoch_F1_prefit.png         # Pre-fit plot
│       ├── Epoch_F1_bestfit.png        # SED plot after fiting
│       ├── Epoch_F1_fit_results.txt    # Fit result
│       ├── Epoch_F1_fit_params.csv     # Fitted parameter values
├── Epoch_F2/, Epoch_F3/, Epoch_Q1/ 
```

**Workflow**:
1. Loads configuration from `{epoch}/run{iteration}/config.yaml`
2. Reads SED data from `{epoch}/{epoch}_sed.ecsv` 
3. Builds ExternalComptonModel
4. Performs SED fitting using Gammapy
5. Saves outputs to `{epoch}/run{iteration}/`:
   - `{epoch}_prefit.png`
   - `{epoch}_bestfit.png`
   - `{epoch}_fit_results.txt`
   - `{epoch}_fit_params.csv`
   - `{epoch}.log`

**Usage**:
```bash
python run_agnpy.py Epoch_F1/run1
```

**`prefitting.py`**: 
- Tests multiple model configurations before performing the full fit. Comparison of different parameter sets.
- Usage: `python prefitting.py Epoch_F1 1`

**`plot_agnpy.py`**:
- Creates final SED plots from fitted parameters.
- Usage: `python plot_agnpy.py Epoch_F1/run1`
