# AI-Based Image Restoration and Super-Resolution using SRGAN

A deep-learning pipeline that combines **CNN-based image denoising** with **4× super-resolution using a Super-Resolution Generative Adversarial Network (SRGAN)**. The goal is to clean degraded low-resolution images and reconstruct sharper, higher-resolution outputs.

## Project Overview

The pipeline has three main components:

1. **Denoising Autoencoder (DAE):** removes synthetic Gaussian noise from low-resolution images.
2. **SRGAN Generator and Discriminator:** reconstructs an image at 4× spatial resolution and learns to produce perceptually realistic details through adversarial training.
3. **VGG19 Perceptual Loss:** compares deep image features to encourage visually convincing results in addition to pixel-level similarity.

### Full Pipeline

```text
Noisy Low-Resolution Image
          |
          v
CNN Denoising Autoencoder
          |
          v
Clean Low-Resolution Image
          |
          v
SRGAN Generator (4× upscaling)
          |
          v
Super-Resolved High-Resolution Image
```

## Key Features

- CNN autoencoder with an encoder–decoder structure and skip connection for denoising.
- SRGAN generator with residual blocks and PixelShuffle upsampling.
- CNN discriminator for adversarial training.
- VGG19-based perceptual loss combined with pixel and adversarial losses.
- Generator pretraining before full GAN training.
- Early stopping and checkpoint saving for model training.
- Evaluation using **PSNR** and **SSIM**, with bicubic interpolation as a baseline.
- Google Colab workflow designed to use a T4 GPU and Google Drive for persistent files.

## Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

Additional libraries: `torchvision`, `scikit-image`, `torchsummary`, and `matplotlib`.

## Dataset

The notebook uses the **DIV2K validation high-resolution image set**. Low-resolution images are generated from high-resolution images using bicubic downsampling.

| Setting | Value |
|---|---|
| Dataset | DIV2K validation set |
| High-resolution patch | 64 × 64 pixels |
| Low-resolution patch | 16 × 16 pixels |
| Upscaling factor | 4× |
| Batch size | 2 |
| Compute environment | Google Colab (T4 GPU recommended) |

> **Note:** The supplied notebook uses the DIV2K validation set to create training patches. For a rigorous benchmark, train on the DIV2K training split and reserve separate images for final evaluation.

## Model Architecture

### 1. CNN Denoising Autoencoder

The autoencoder cleans the low-resolution image before super-resolution.

- **Encoder:** convolutional layers, ReLU activation, and max pooling.
- **Decoder:** transposed convolution and convolutional layers.
- **Skip connection:** combines early encoder features with decoder features to retain spatial details.
- **Training objective:** mean squared error (MSE) between the denoised output and the clean low-resolution target.
- **Noise:** Gaussian noise is added to the input during training.

### 2. SRGAN Generator

The generator maps a 16 × 16 low-resolution image to a 64 × 64 super-resolved image.

- Initial 9 × 9 convolution for feature extraction.
- Six residual blocks by default.
- Two PixelShuffle upsampling stages, each scaling spatial dimensions by 2.
- Final convolution and `tanh` activation.

### 3. Discriminator

A VGG-style convolutional network predicts whether an input high-resolution image is real or generated. It uses strided convolutions, batch normalization, LeakyReLU activations, and a final sigmoid output.

### 4. Combined Generator Loss

The generator combines three objectives:

| Loss | Weight | Purpose |
|---|---:|---|
| Pixel MSE | 1.0 | Encourages pixel-level similarity to the target |
| Adversarial loss | 0.001 | Encourages realistic-looking details |
| VGG19 perceptual loss | 0.006 | Encourages similarity in deep visual features |

```text
L_total = L_pixel + 0.001 × L_adversarial + 0.006 × L_perceptual
```

The VGG19 feature extractor is frozen and is used only to calculate perceptual loss.

## Training Workflow

1. Install dependencies and select CUDA when available.
2. Download or restore the DIV2K validation images.
3. Create paired low-resolution and high-resolution patches.
4. Train the denoising autoencoder with early stopping.
5. Pretrain the SRGAN generator using pixel MSE.
6. Train the generator and discriminator adversarially using the combined loss.
7. Save model weights and generate qualitative and quantitative evaluation outputs.

## Getting Started

### Requirements

- Python 3
- PyTorch and torchvision
- OpenCV
- scikit-image
- NumPy
- Matplotlib
- Google Colab with GPU runtime recommended
- Google Drive for persistent model checkpoints when using the supplied notebook workflow

### Run in Google Colab

1. Open the project notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run the dependency-installation cell.
4. Mount Google Drive when prompted.
5. Run the notebook cells in order.

The notebook saves checkpoints and visualizations in:

```text
/content/drive/MyDrive/SRGAN_Project/
```

Expected checkpoint filenames:

```text
srgan_denoiser.pth
srgan_generator.pth
srgan_discriminator.pth
```

### Install dependencies locally

```bash
pip install torch torchvision opencv-python-headless scikit-image torchsummary numpy matplotlib
```

The provided notebook includes Google Colab-specific paths and shell commands, so those parts need adaptation before running the workflow locally.

## Evaluation

The notebook compares SRGAN outputs with bicubic interpolation using:

- **PSNR (Peak Signal-to-Noise Ratio):** measures pixel-level reconstruction error; higher is generally better.
- **SSIM (Structural Similarity Index):** measures structural similarity; values closer to 1 indicate greater similarity.

The notebook also produces visual comparisons of the original low-resolution input, denoised input, SRGAN output, and high-resolution reference.

**Quantitative results:** No final metric values are included in the supplied project text. Run the evaluation cells to generate the actual PSNR and SSIM values before reporting them.

## Generated Outputs

The notebook is designed to save these visualizations to Google Drive:

- `dataset_samples.png` — low-resolution inputs and high-resolution references.
- `dae_results.png` — denoising autoencoder comparisons.
- `full_pipeline_results.png` — complete pipeline outputs and metric comparisons.

## Project Structure

A suggested repository structure is:

```text
.
├── README.md
├── notebook.ipynb
├── dataset/                 # Downloaded data; usually keep out of Git
├── outputs/
│   ├── dataset_samples.png
│   ├── dae_results.png
│   └── full_pipeline_results.png
└── checkpoints/             # Model weights; usually keep out of Git
```

The actual notebook currently stores data, checkpoints, and output images in Google Drive. Adjust this structure to match the files included in your repository.

## Important Notes

- SRGAN is designed to improve perceptual sharpness, so it may not outperform bicubic interpolation on PSNR alone.
- The supplied notebook evaluates four sample images in its final visualization. Use a larger held-out test set for reliable benchmark claims.
- Model checkpoints and datasets can be large; consider keeping them out of Git and documenting how to obtain them.
- The DIV2K dataset has its own usage terms. Review those terms before redistributing the images.

## Reference

Ledig et al., **“Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network”** (2017).

---

*Developed as a deep-learning project in image restoration and super-resolution.*
