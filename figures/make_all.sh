#!/bin/bash
# Regenerate all figures of the paper into figures/output/ (needs numpy and matplotlib).
set -e
cd "$(dirname "$0")"
for s in fig2_fig4_learning.py fig3_v2i_communication.py fig5_origin_inference.py \
         appendix_e1_seeds.py appendix_cpq_lambda_paired.py appendix_e3_signatures.py; do
    python "$s"
done
