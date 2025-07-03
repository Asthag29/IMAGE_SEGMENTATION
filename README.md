# 🐟 FISH_SEGMENTATION

A deep learning project to segment fish rib structures from images using a simplified U-Net architecture.

---

## 📦 Project Structure ##

```plaintext
fish_project/
├── fisherv/                   # Additional utilities or archives
├── notebooks/                 # Jupyter notebooks for experimentation
├── src/                       # Source code (extendable)
├── data/                      # Datasets
│   ├── Fish_Dataset/
│   ├── A_Fish_Dataset/
│   └── NA_Fish_Dataset/
├── model_paths/               # Trained model checkpoints and metadata
│   ├── best_fish_seqm...
│   ├── train/
├── reports/                   # Output reports
│   ├── fish-test.png          # Sample mask generated on a test sample
│   └── loss_graph.png         # Training & validation loss curve
├── train/                     # Training pipeline code
│   ├── api.py
│   ├── dataloaders.py
│   ├── seg_model.py
│   └── train.py
├── .gitignore
├── .python-version
├── README.md
├── requirements.txt


## ⚙️ Training Details

- **Loss Function**: `CrossEntropyLoss`  
  Chosen to directly optimize for binary segmentation, ensuring each mask pixel is predicted as either 0 or 1.
- **Optimizer**: Adam optimizer with a learning rate of `0.001`.
- **Batch Size**: 96  
  Selected to **optimally utilize the large memory (VRAM)** available on the NVIDIA RTX 8090 GPU.  
  A larger batch size speeds up convergence and stabilizes gradient updates, without causing out-of-memory errors.
- **Hardware**: Training performed on an **NVIDIA RTX 4090 GPU**  
  The powerful GPU significantly reduced training time (around 30 minutes) and made it feasible to train with higher batch sizes.
- **Validation Strategy**:  
  Since masks were not provided for the test set, we created a dedicated **validation dataset** from the labeled data to test and fine-tune the model.

---

## 📊 Reports & Results

Output results and plots are stored in the `reports/` directory:

- `fish-test.png`: Example of a segmentation mask generated on a test sample.
- `loss_graph.png`: Plot showing the training and validation loss curves over epochs.

These artifacts help evaluate model performance and monitor trends like overfitting or underfitting.

---

## 📁 Model Checkpoints

All trained model files, along with metadata (e.g., number of epochs, learning rate, optimizer state), are saved under the `model_paths/` folder.  
This enables reuse, fine-tuning, and reproducible experiments without retraining from scratch.

---

## 🧪 Experimentation

The `notebooks/` directory contains Jupyter notebooks used for:

- Exploratory data analysis
- Visualizing intermediate outputs
- Experimenting with model variations and hyperparameters

These notebooks act as a sandbox for rapid experimentation and visualization.

---

## 📜 Installation

Install all required dependencies:

```bash```
pip install -r requirements.txt

