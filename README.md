# LSE-conMod

LSE-conMod uses large-sample emulators (LSEs) to calibrate and regionalize parameters for conceptual hydrological models. The companion study evaluates Global Optimization And Localized Selection (GOALS) with the HBV model across 531 CAMELS basins.

## Code and data for the paper

Use the [versioned Zenodo release, v1.0.0](https://doi.org/10.5281/zenodo.23873715) for the study workflows. Download `LSE_conMod_public_release.zip`, extract it, and follow its `README.md` and `NOTEBOOK_GUIDE.md`. The archive includes five-fold spatial cross-validation, generation and local selection of regionalized parameter candidates, the single-site emulator (SSE) comparison, selected analysis notebooks, and the processed `camels_data/` inputs.

This GitHub repository contains the earlier HBV calibration experiments. Its scripts and settings differ from the archived implementation, and its `data/` directory contains parameter bounds, basin metadata, and preparation notebooks. Use the code and configurations from the same snapshot when running the study workflows.

## GitHub repository layout

| Directory | Contents |
| --- | --- |
| [`HBV/`](HBV/) | HBV model implementations |
| [`LSEexp/`](LSEexp/) | LSE calibration scripts, experiment TOML files, and PBS examples |
| [`SSEexp/`](SSEexp/) | SSE calibration and optimization scripts and PBS examples |
| [`auxiliary/`](auxiliary/) | Data loading, model execution, evaluation, and optimization helpers |
| [`data/`](data/) | HBV parameter bounds, CAMELS basin metadata, and preparation notebooks |

## Running the GitHub experiments

Install the direct Python dependencies with `python -m pip install -r requirements.txt`. The workflows also require the MO-ASMO modules `sampling`, `gp`, and `NSGA2` on the Python import path. The SSE workflow additionally imports `mo_evaluation` from the research support code included in the Zenodo archive. See [NOTICE](NOTICE) for provenance.

The TOML files and PBS scripts retain the original NCAR HPC settings. Point `inpath_data` to the extracted `camels_data/` directory and check the parameter bounds, output, external support module, and PBS settings for your environment. Run the scripts from their own experiment directories so the relative imports resolve.

For example, from the repository root, initialize the historical HBV-KM LSE experiment with the following command:

```sh
cd LSEexp
python main_LSEworkflow_HBVKM.py config_LSEexp7.toml 0 1
```

The last two arguments specify the iteration and CPU count. Subsequent iterations use results from earlier iterations. The historical `SSEexp1` configuration reuses the LSE iteration-zero simulations via `external_iter0_path`.

## Citation

Tang, G., Wood, A. W., Song, Y., Farahani, M. A., & Shen, C. (in press). Calibration, Regionalization, and Locally-Informed Selection of Hydrological Model Parameters Using AI-Based Large-Sample Emulators. *Water Resources Research*.

For the archived code and processed inputs, cite Tang, G., & Wood, A. (2026). *LSE-conMod: Code and Data for Hydrological Parameter Calibration and Regionalization* (v1.0.0). Zenodo. [10.5281/zenodo.23873715](https://doi.org/10.5281/zenodo.23873715).

When using the CAMELS data, also cite the [time series](https://doi.org/10.5065/D6MW2F4D) and [basin attributes](https://doi.org/10.5065/D6G73C3Q). Machine-readable citation metadata are in [CITATION.cff](CITATION.cff).

## Licensing and attribution

See [LICENSE](LICENSE) for the proposed license for original project software and [NOTICE](NOTICE) for the scope and third-party material.
