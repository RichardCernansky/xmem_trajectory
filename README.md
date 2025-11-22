# XMem Trajectory Extension

## Folder Structure

```plaintext
XMem/
 ├─ xmem/                
 │    └─ ...              # Original XMem implementation
 ├─ traj/                 # Trajectory prediction extension
 │    ├─ predictor.py     # Wrapper that uses XMem encoders + memory
 │    ├─ head.py          # Small GRU/MLP head
 │    └─ datamodules.py   # Lightweight nuScenes sequence loader (stub)
 └─ train_traj.py         # Training script

REQUIREMENTS
# PyTorch with CUDA 12.4
--index-url https://download.pytorch.org/whl/cu124
torch==2.4.1
torchvision==0.20.1

# Core dependencies with version pins
numpy==1.26.4
opencv-python==4.10.0.84
pandas==2.2.3

# Scientific computing
scipy
scikit-learn
matplotlib
pillow
shapely
pyquaternion

# nuScenes
nuscenes-devkit==1.2.0

# OpenMMLab (install mmcv via mim separately)
mmengine==0.10.7
openmim==0.3.9

# Utilities
tqdm
rich
pyyaml
termcolor






softmax edit xmem
xmem edit memoery manager line 222 gv unsqueeze


clone repo
clone checkpoint
requirements 
get blob
run program

#INDEX script:
echo "== Clone =="
git clone --recurse-submodules https://github.com/RichardCernansky/xmem_trajectory.git app
cd app

echo "== Python deps =="
python -m pip install -U pip
pip install -U nuscenes-devkit numpy pyquaternion

echo "== Mounts =="
ROOT="${{inputs.ds_root}}"
DATA="${ROOT}/nuscenes"
OUT_DIR="${ROOT}/indexes/run-get-30000"
echo "ROOT=${ROOT}"
echo "DATA=${DATA}"
echo "OUT_DIR=${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo "== Peek dataset =="
ls -al "${DATA}" | head
find "${DATA}" -maxdepth 2 -type f | head -n 10

echo "== Build index =="
time python -m index_nuscenes.build_index \
  --dataroot "${DATA}" \
  --n_total 30000 \
  --train_path "${OUT_DIR}/train_agents_index.pkl" \
  --val_path   "${OUT_DIR}/val_agents_index.pkl" \
  2>&1 | tee "${OUT_DIR}/run.log"

echo "== Output preview =="
ls -al "${OUT_DIR}" | head
echo "== DONE =="
date


#TRAINER script:
echo "== Clone =="
git clone --recurse-submodules https://github.com/RichardCernansky/xmem_trajectory.git app
cd app 
curl -L -o checkpoints/xmem/XMem-s012.pth \
    https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth
ls -lh XMem/checkpoints/XMem-s012.pth

echo "== Paths =="
ROOT="${{inputs.ds_root}}"
DATA="${ROOT}/nuscenes"
TRAIN_IDX="${ROOT}/indexes/run-get-1000/train_agents_index.pkl"
VAL_IDX="${ROOT}/indexes/run-get-1000/val_agents_index.pkl"
CKPTS="${ROOT}/checkpoints"
echo "DATA=$DATA"; echo "TRAIN_IDX=$TRAIN_IDX"; echo "VAL_IDX=$VAL_IDX"; echo "CKPTS=$CKPTS"

echo "== Train =="
python -m trainer.trainer \
  --model_name mem_v1 \
  --version v1.0-trainval \
  --dataroot "$DATA" \
  --train_index "$TRAIN_IDX" \
  --val_index   "$VAL_IDX" \
  --checkpoints_dir "$CKPTS" \
  --epochs 20 \
  --resume


python -m trainer.trainer   --model_name mem_v1   --version v1.0-trainval   --dataroot /home/cernanskyr/v1.0-mini   --train_index ./data/indexes/train_agents_index.pkl   --val_index  ./data/indexes/val_agents_index.pkl   --checkpoints_dir ./checkpoints   --epochs 10 

Local index build commands
python -m index_nuscenes.build_index --train_path ./data/indexes/train_agents_index.pkl --val_path ./data/indexes/val_agents_index.pkl --dataroot /home/cernanskyr/v1.0-mini --n_total 500



IMPORTANT NOTES
- stiahnut nuscenes
- rozdellit xmem wrapper na train adn val
-GT vs pred_probs
softmax edit xmem
xmem edit memoery manager line 222 gv unsqueeze
# TODO: SOLVE NORMALIZARION, update optimizer optimizer, mask=union detachment in predictor, decide on strategy of xmem pre-training


Xmem predictor training
epoch 0: train={'mask_loss': 0.6472291350364685}, val={'mask_loss': 0.7567994594573975}                                                                                                                                                                                                 
epoch 1: train={'mask_loss': 0.5467812418937683}, val={'mask_loss': 0.7568157315254211}                                                                                                                                                                                                 
epoch 2: train={'mask_loss': 0.7769875526428223}, val={'mask_loss': 0.7568175196647644}                                                                                                                                                                                                 
epoch 3: train={'mask_loss': 0.7351729273796082}, val={'mask_loss': 0.7568275928497314}                                                                                                                                                                                                 
epoch 4: train={'mask_loss': 0.6627680659294128}, val={'mask_loss': 0.7568188309669495}                                                                                                                                                                                                 
epoch 5: train={'mask_loss': 0.5467884540557861}, val={'mask_loss': 0.7568191289901733}                                                                                                                                                                                                 
epoch 6: train={'mask_loss': 0.4553024470806122}, val={'mask_loss': 0.7568931579589844}                                                                                                                                                                                                 
epoch 7: train={'mask_loss': 0.5361988544464111}, val={'mask_loss': 0.7568467855453491}                                                                                                                                                                                                 
epoch 8: train={'mask_loss': 0.7973802089691162}, val={'mask_loss': 0.7568185925483704}                                                                                                                                                                                                 
epoch 9: train={'mask_loss': 0.8118299841880798}, val={'mask_loss': 0.7565058469772339}                                                                                                                                                                                                 
epoch 10: train={'mask_loss': 0.8118299841880798}, val={'mask_loss': 0.7565058469772339}                                                                                                                                                                                                
epoch 11: train={'mask_loss': 0.7588531374931335}, val={'mask_loss': 0.7565058469772339}   



