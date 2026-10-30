#!/bin/bash
# Rebuild every figure (data, PDFs, PeleC-only variants, PNG previews)
set -e
cd "$(dirname "$0")"
/lustre/orion/cfd154/scratch/sbryngelson/viz-venv/bin/python -I make_data.py > /dev/null
FIGS="scaling_slide lockstep tts fixes amr_only coverage interface_error rhs_models convergence"
for f in $FIGS; do latexmk -pdf -silent -interaction=nonstopmode $f.tex > /dev/null; done
for f in scaling_slide tts fixes amr_only; do
  latexmk -pdf -silent -interaction=nonstopmode -jobname=${f}_pelec -usepretex='\def\noamrex{}' $f.tex > /dev/null
  FIGS="$FIGS ${f}_pelec"
done
mkdir -p png
for f in $FIGS; do pdftoppm -png -r 200 -singlefile $f.pdf png/$f; done
echo "built: $FIGS"
