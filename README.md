**Satellite Water U-Net — Final Project Submission**

**Project Purpose:** This project investigates whether a convolutional segmentation model (U-Net) can accurately identify water bodies in high-resolution satellite imagery when portions of the image are occluded by synthetic clouds. Cloud cover is a common problem in remote sensing; the goal is to measure the model's ability to leverage spatial context to recover water masks from partially missing information.

**Dataset & Creation:**
- Source: Satellite Water Bodies Dataset (publicly available; original source referenced in the code). A small subset of the dataset is included under `data/images` and `data/masks` for demonstration and reproducibility.
- How it was created: original RGB satellite images and binary water masks were paired by filename stem. During training and evaluation the pipeline generates synthetic cloud occlusions (see `swu/masking.py`) by overlaying random opaque patches on input images. This simulates varying cloud coverage levels and tests robustness to missing input regions.

**Files you should look at:**
- Model implementation: `swu/unet.py` (production). A convenience wrapper is provided at `models.py`.
- Dataset loader: `swu/dataset.py` (production). A top-level `dataset.py` wrapper is provided so notebooks can import `WaterDataset` directly.
- Training script: `train.py` (full training loop). A convenience entry `train_models.py` is provided for programmatic invocation.
- Demo notebooks: `notebooks/data_demo.ipynb` (dataset examples) and `notebooks/predictions.ipynb` (inference + evaluation visualizations).

**How to train**
1. Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

2. Run training (example):

```bash
python3 train.py \
	--image-dir data/images \
	--mask-dir data/masks \
	--epochs 20 \
	--batch-size 8 \
	--lr 1e-3 \
	--output-dir outputs
```

3. Programmatic training wrapper (from Python):

```python
from train_models import train_from_argv
train_from_argv(["--image-dir","data/images","--mask-dir","data/masks","--epochs","5"])
```

Training produces per-epoch checkpoints in `outputs/` (e.g. `unet_epoch1.pth`, `unet_best.pth`) and a saved loss curve `outputs/loss_curve.png` when matplotlib is available.

**Results & Metrics**
- Saved model weights: `outputs/unet_best.pth` (best validation loss) and per-epoch checkpoints in `outputs/`.
- Evaluation metrics reported by the training script and notebooks:
	- Intersection over Union (IoU)
	- Dice Score (F1-like score for segmentation)
	- Pixel Accuracy

Example final test scores are printed at the end of `train.py` and recorded in the notebook evaluation outputs. Use `notebooks/predictions.ipynb` to reproduce evaluation plots and numeric summaries.

**Visualization of Predictions**
- `notebooks/data_demo.ipynb` shows dataset examples (original, masked input, and target mask).
- `notebooks/predictions.ipynb` runs the trained model on held-out samples and produces overlay plots (predicted mask vs ground truth), as well as IoU/Dice histograms. Open these with Jupyter or JupyterLab.

**Limitations & Discussion**
- Synthetic clouds differ from real atmospheric scattering and thin clouds; models trained on synthetic occlusions may not generalize perfectly to real cloud cover.
- Dataset size in this repository is a small demonstration subset. For production-quality results, train on the full dataset on a GPU-equipped cluster (Talapas or similar).
- U-Net trained with standard BCE-with-logits loss handles binary masks well, but more advanced losses (e.g., boundary-aware losses or focal variants) may improve performance on small water bodies.
- Geographical and seasonal bias: the dataset comes from a limited set of scenes and may not generalize across sensor types, seasons, or geographic regions.

**Paths & Where to find things**
- Example data (in this repo): `data/images/` and `data/masks/`.
- Model code: `models.py` (wrapper) and `swu/unet.py` (implementation).
- Dataset class: `dataset.py` (wrapper) and `swu/dataset.py` (implementation).
- Training script: `train.py` and `train_models.py` (programmatic wrapper).
- Trained weights: `outputs/unet_best.pth` and `outputs/unet_epoch*.pth`.

If you store full dataset files or final model weights on Talapas or GitHub releases, include direct paths or download instructions here (replace placeholders):

 - Dataset (full): /path/on/talapas/datasets/satellite-water-bodies/
 - Best model weights: /path/on/talapas/outputs/unet_best.pth

**Reproducibility notes**
- Random seed for dataset splits is fixed where applicable to allow reproducible splits.
- Use a GPU for training larger models or the complete dataset. Adjust `--device` in `train.py`.

If you want, I can (1) run the training locally on a small subset to produce example outputs, (2) update the notebooks to import the top-level `dataset.py` instead of `swu.dataset`, or (3) prepare a short submission checklist and ZIP for upload to Talapas.