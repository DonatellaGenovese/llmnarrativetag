#!/bin/bash
# Step 3 on Leonardo: cache ParT's 128-d class token for every TopLandscape jet,
# from the JetClass pretraining (models/ParT_kin.pt), frozen. It does not use the
# fine-tune, so it can run alongside part1_finetune.sbatch.
#
#   cd ~/llmnarrativetag && sbatch scripts/slurm/dump_embeddings.sbatch
#
# Writes surrogate/outputs/top/embeddings/xcls_{train,val,test}.npz, which is
# where anomaly.baselines and anomaly.transfer read them from.

#SBATCH --job-name=part-embed
#SBATCH --account=IscrC_XJET-HEP
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$HOME/llmnarrativetag}"
cd "$REPO/particle_transformer"
source env.sh
source .venv/bin/activate

echo "### $(date '+%F %T')  node $(hostname)  job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU visible to torch')"

# The script steps into particle_transformer/ on its own and resolves the
# dataset and checkpoint from there; the output path defaults to the repo's
# surrogate/outputs/top/embeddings/.
for split in test val train; do
  echo "### $split  $(date '+%T')"
  python "$REPO/scripts/dump_part_embeddings.py" --split "$split" --gpus 0
done

# Row alignment with the dataset, which the anomaly code relies on (it joins
# the embeddings to the feature table by row).
python - <<'EOF'
import numpy as np, pyarrow.parquet as pq
for s in ("train", "val", "test"):
    e = np.load(f"../surrogate/outputs/top/embeddings/xcls_{s}.npz")
    lab = pq.read_table(f"datasets/TopLandscape/{s}_file.parquet", columns=["label"]).column("label").to_numpy()
    # dataset label is 1 = top; the npz uses the feature table's 0 = top.
    ok = len(e["x"]) == len(lab) and (e["label"] == 1 - lab).all()
    print(f"{s:5s} {len(e['x']):>9,} jets  x {e['x'].shape}  aligned with dataset: {ok}")
    assert ok, f"{s}: embeddings not row-aligned with the dataset"
EOF

echo "### done  $(date '+%F %T')"
ls -lh "$REPO/surrogate/outputs/top/embeddings/"
