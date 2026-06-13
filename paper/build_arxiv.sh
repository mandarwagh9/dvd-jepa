#!/usr/bin/env bash
# Bundle the arXiv submission tarball from the canonical paper/ sources.
# Usage:  bash paper/build_arxiv.sh
set -e
cd "$(dirname "$0")"
tar czf dvd-jepa-arxiv.tar.gz \
  main.tex \
  metrics.tex \
  fig/fig_collapse.pdf \
  fig/fig_horizon.pdf \
  fig/fig_anomaly.pdf \
  fig/fig_filmstrip.png
echo "wrote paper/dvd-jepa-arxiv.tar.gz  (upload this at https://arxiv.org/submit)"
tar tzf dvd-jepa-arxiv.tar.gz
