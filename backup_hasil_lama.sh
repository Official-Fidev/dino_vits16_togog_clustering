#!/bin/bash
# Pindahkan hasil lama (run dengan fallback ViT-B/16, fitur 768 dimensi)
# supaya tidak tercampur dengan hasil ViT-S/16 yang baru.
set -e
BK="outputs_lama_vitb16_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BK"
for d in features embeddings clusters plots results history; do
  [ -d "outputs/$d" ] && mv "outputs/$d" "$BK/"
done
[ -f docs/skripsi/PERBANDINGAN_SESSION.md ] && cp docs/skripsi/PERBANDINGAN_SESSION.md "$BK/"
echo "Hasil lama dipindahkan ke $BK"
