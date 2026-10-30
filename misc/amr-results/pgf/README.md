# Sponsor figures (pgfplots)

Rebuild after new results land:

    /lustre/orion/cfd154/scratch/sbryngelson/viz-venv/bin/python -I make_data.py
    for f in scaling_slide lockstep tts fixes convergence; do latexmk -pdf $f.tex; done
    # PeleC-only sponsor variants (no AMReX CNS), written as *_pelec.pdf
    for f in scaling_slide tts fixes; do latexmk -pdf -jobname=${f}_pelec -usepretex="\\def\\noamrex{}" $f.tex; done

- `scaling_slide`: subcycled MFC vs AMReX CNS vs PeleC: efficiency, AMR cost added beyond 8 GPUs, s/step.
- `lockstep`: lock-step efficiency and s/step, MFC vs AMReX CNS.
- `tts`: wall time to the same physical time at 128 GPUs.
- `fixes`: what each fix bought, MFC subcycled at 64 GPUs; rows for the subcycling driver, swap packing and tile
  clustering appear once `mfc-simple-v3/results.csv` has them.
- `convergence`: subcycled minus lock-step solution vs level-0 dt (minmod vs quartic).

Data: `mfc-simple-v2/results.csv` (+ `mfc-simple-v3/results.csv`) and `progress/data.json`. Style: `style.tex`.
