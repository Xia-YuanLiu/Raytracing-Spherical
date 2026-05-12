# Generated Outputs

This directory contains generated figures and data grouped by experiment or paper section.

- `junction_atlas/`: RN/RN-dS static junction atlas output from `scripts/generate_junction_atlas.py --preset quick`. Contains the atlas manifest, phase maps, selected case artifacts, and report-facing figure copies.
- `junction_atlas_schwarzschild_reference/`: Schwarzschild reference output from `scripts/generate_junction_atlas.py --preset schwarzschild-reference`. Contains Fig. 3 through Fig. 8 reproduction artifacts.
- `rn_junction_sweep/`: RN static junction sweep outputs from `scripts/generate_rn_junction_images.py`.
- `schwarzschild_fig5/`: Schwarzschild thin-disk Fig. 5 profile and image outputs from `scripts/generate_fig5_profiles.py`.
- `lqg_fig3/`: LQG Fig. 3 profile, image, and ring-edge outputs from `scripts/generate_lqg_fig3_profiles.py`.
- `static_junction/fig3_fig4/`: Static junction Fig. 3/Fig. 4 reproduction outputs from `scripts/generate_static_junction_fig3_fig4.py`.

Large generated atlas directories should usually remain untracked unless a task explicitly asks to commit generated figures.
