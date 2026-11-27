#!/bin/bash
# Menjalankan ulang pipeline dengan ViT-S/16 (timm, 384 dimensi).
# Log: run_ulang_vits16.log   Status akhir: run_ulang_vits16.status
cd "$(dirname "$(readlink -f "$0")")"
LOG="run_ulang_vits16.log"; ST="run_ulang_vits16.status"
echo "RUNNING" > "$ST"
# Tampilkan keluaran di terminal DAN simpan ke berkas log sekaligus.
exec > >(tee "$LOG") 2>&1
set -x
echo "START $(date)"

# 1. Cari Python yang sudah punya torch + umap + hdbscan + optuna
PY=""
for c in ./dino-env/bin/python ../dino_model_backup/dino_model/dino-env/bin/python \
         "$HOME"/miniconda3/envs/*/bin/python "$HOME"/anaconda3/envs/*/bin/python \
         "$HOME"/miniconda3/bin/python "$HOME"/anaconda3/bin/python \
         "$HOME"/.venv/bin/python "$(command -v python3)"; do
  [ -x "$c" ] || continue
  if "$c" -c "import torch, umap, hdbscan, optuna, sklearn, pandas, seaborn" 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Tidak ada Python dengan dependensi lengkap; menjalankan setup.sh"
  bash setup.sh || { echo "FAILED setup" > "$ST"; exit 1; }
  PY=./dino-env/bin/python
fi
echo "PYTHON: $PY"
"$PY" -m pip install timm || { echo "FAILED pip timm" > "$ST"; exit 1; }
"$PY" -c "import torch;print('cuda',torch.cuda.is_available())"
"$PY" -c "import timm;m=timm.create_model('vit_small_patch16_224.dino',pretrained=True,num_classes=0);print('feature_dim',m.num_features)" \
  || { echo "FAILED load model" > "$ST"; exit 1; }

# 2. Amankan hasil lama (ViT-B/16)
bash backup_hasil_lama.sh

set -e
trap 'echo "FAILED line $LINENO" >> "$ST"' ERR
# 3. Pipeline (sama dengan run_all.sh)
"$PY" preprocess.py
"$PY" extract_features.py
for i in $(seq 1 10); do
  echo "=== ITERASI $i ==="
  "$PY" src/dim_reducer.py --features outputs/features/resized_256x256/features.npy --output-dir outputs/embeddings/resized_256x256 --method umap --n-components 10 --n-neighbors 30 --min-dist 0.0 --metric cosine
  "$PY" src/dim_reducer.py --features outputs/features/resized_256x256/features.npy --output-dir outputs/embeddings/resized_256x256 --method umap --n-components 2 --n-neighbors 30 --min-dist 0.0 --metric cosine
  "$PY" src/auto_tune.py --embeddings outputs/embeddings/resized_256x256/embeddings_umap_10d.npy --output outputs/results/best_params.json --n-trials 30
  "$PY" src/clustering.py --embeddings outputs/embeddings/resized_256x256/embeddings_umap_10d.npy --embeddings-2d outputs/embeddings/resized_256x256/embeddings_umap_2d.npy --output-dir outputs/clusters/resized_256x256 --method hdbscan --params-file outputs/results/best_params.json --save-plots
  "$PY" visualize_clusters.py --embeddings outputs/embeddings/resized_256x256/embeddings_umap_2d.npy --clusters outputs/clusters/resized_256x256/cluster_assignments_hdbscan.csv --metadata outputs/features/resized_256x256/metadata.json --output-dir outputs/plots/resized_256x256 --raw-images data/raw/resized_datasets/resized_256x256 --samples 15
  "$PY" visualize_comparison.py --clusters outputs/clusters/resized_256x256/cluster_assignments_hdbscan.csv --metadata outputs/features/resized_256x256/metadata.json --raw-images data/raw/resized_datasets/resized_256x256 --output-dir outputs/plots/resized_256x256 --samples 6
  H="outputs/history/run_iter${i}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$H"
  cp -r outputs/clusters outputs/plots outputs/results "$H/"
  echo "ITERASI $i SELESAI" >> "$ST"
done
"$PY" generate_report.py
echo "DONE $(date)"
echo "DONE" >> "$ST"
