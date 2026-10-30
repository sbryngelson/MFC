# Sponsor figures (pgfplots)

Rebuild after new results land:

    /lustre/orion/cfd154/scratch/sbryngelson/viz-venv/bin/python -I make_data.py
    for f in scaling_slide lockstep tts fixes amr_only coverage interface_error rhs_models convergence; do latexmk -pdf $f.tex; done
    # PeleC-only sponsor variants (no AMReX CNS), written as *_pelec.pdf
    for f in scaling_slide tts fixes amr_only; do latexmk -pdf -jobname=${f}_pelec -usepretex="\\def\\noamrex{}" $f.tex; done

- `scaling_slide`: subcycled MFC vs AMReX CNS vs PeleC: efficiency, AMR cost added beyond 8 GPUs, s/step.
- `lockstep`: lock-step efficiency and s/step, MFC vs AMReX CNS.
- `tts`: wall time to the same physical time at 128 GPUs.
- `fixes`: what each fix bought, MFC subcycled at 64 GPUs; rows for the subcycling driver, swap packing and tile
  clustering appear once `mfc-simple-v3/results.csv` has them.
- `amr_only`: AMR-only s/step (each code's own RHS excluded), subcycled and lock-step.
- `coverage`: refined share of the domain, MFC default vs tile clustering vs AMReX CNS / PeleC.
- `interface_error`: L1 error in the coarse-fine band relative to the uniform fine grid, minmod vs quartic.
- `rhs_models`: RHS ns/cell for the same numerics in HIP, C+OpenMP and Fortran+OpenMP variants.
- `convergence`: subcycled minus lock-step solution vs level-0 dt (minmod vs quartic).

Data: `mfc-simple-v2/results.csv` (+ `mfc-simple-v3/results.csv`) and `progress/data.json`. Style: `style.tex`.
