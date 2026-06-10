**Satellite Water U-Net — Final Project Submission**

**Project Purpose:** This project investigates whether a convolutional segmentation model (U-Net) can accurately identify water bodies in high-resolution satellite imagery when portions of the image are occluded by synthetic clouds. Cloud cover is a common problem in remote sensing; the goal is to measure the model's ability to leverage spatial context to recover water masks from partially missing information.

**Dataset & Creation:**
- Source: Satellite Water Bodies Dataset (publicly available; original source referenced in the code). A small subset of the dataset is included under `data/images` and `data/masks` for demonstration and reproducibility.
- How it was created: original RGB satellite images and binary water masks were paired by filename stem. During training and evaluation the pipeline generates synthetic cloud occlusions (see `swu/masking.py`) by overlaying random opaque patches on input images. This simulates varying cloud coverage levels and tests robustness to missing input regions.

**Key files:**
- `swu/unet.py` — U-Net model implementation.
- `swu/dataset.py` — dataset and preprocessing utilities.
- `swu/masking.py` — synthetic cloud masking pipeline.
- `train.py` — training and evaluation loop (command-line entry).
- `models.py`, `dataset.py`, `train_models.py` — top-level wrappers for easier imports and programmatic use.
- Notebooks: `notebooks/data_demo.ipynb` and `notebooks/predictions.ipynb`.

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

Example test scores are printed at the end of `train.py` and recorded in the notebook outputs. Use `notebooks/predictions.ipynb` to reproduce evaluation plots and numeric summaries.

**Visualization of Predictions**
- `notebooks/data_demo.ipynb` shows dataset examples (original, masked input, and target mask).
- `notebooks/predictions.ipynb` runs the trained model on held-out samples and produces overlay plots (predicted mask vs ground truth), as well as IoU/Dice histograms. Open these with Jupyter or JupyterLab.

**Limitations & Discussion**
- Synthetic clouds differ from real atmospheric scattering and thin clouds; models trained on synthetic occlusions may not generalize perfectly to real cloud cover.
- Dataset size in this repository is a small demonstration subset. For production-quality results, train on the full dataset on a GPU-equipped cluster (Talapas or similar).
- U-Net trained with standard BCE-with-logits loss handles binary masks well, but more advanced losses (e.g., boundary-aware losses or focal variants) may improve performance on small water bodies.
- Geographical and seasonal bias: the dataset comes from a limited set of scenes and may not generalize across sensor types, seasons, or geographic regions.

**Paths & Where to find things**
<<<<<<< HEAD
=======

If full dataset files or final model weights are stored on Talapas or GitHub releases, include direct paths or download instructions here (replace placeholders):


**Reproducibility notes**
>>>>>>> 14b9ce7 (Final project submission: finalize README, add wrappers and notebooks edits)
If full dataset files or final model weights are stored on Talapas or GitHub releases, include direct paths or download instructions here (replace placeholders):

- Dataset (full): /path/on/talapas/datasets/satellite-water-bodies/
- Best model weights: /path/on/talapas/outputs/unet_best.pth

**Reproducibility notes**
- Random seed for dataset splits is fixed where applicable to allow reproducible splits.
- GPU recommended for training the full dataset; adjust `--device` in `train.py` as needed.

**Contact / Submission**
For grading or questions, submit this repository and include the Talapas paths or GitHub release links to any large files.
