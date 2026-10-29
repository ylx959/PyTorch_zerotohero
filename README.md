# Deep Learning (PyTorch)

PyTorch deep learning notes, following [Learn PyTorch](https://www.learnpytorch.io/).

## My Progress

| # | Section | Core Skill Built | My Notebook | Status |
|---|---------|------------------|-------------|--------|
| `00` | **PyTorch Fundamentals** | Tensors, dtypes, shapes, matrix multiplication, indexing, reproducibility, GPU operations | 📓 [View](00-pytorch-fundamentals/tensors.ipynb) · [Exercises](00-pytorch-fundamentals/exercise.ipynb) | ✅ Done |
| `01` | **PyTorch Workflow** | The end-to-end loop: data → model → loss → train → evaluate → save | 📓 [View](01-pytorch-workflow/workflow.ipynb) | ✅ Done |
| `02` | **Neural Network Classification** | Binary & multi-class classification, non-linear activation functions, evaluation metrics | 📓 [View](02-pytorch-classification/classification.ipynb) | ✅ Done |
| `03` | **Computer Vision** | CNNs, convolutional layers, pooling, feature maps | — | ⬜ Not started |
| `04` | **Custom Datasets** | Loading real-world image data, custom `Dataset` classes, data augmentation | — | ⬜ Not started |
| `05` | **Going Modular** | Turning notebook code into reusable Python scripts | — | ⬜ Not started |
| `06` | **Transfer Learning** | Fine-tuning pretrained models on new data | — | ⬜ Not started |
| `07` | **Experiment Tracking** | Tracking and comparing experiments with TensorBoard | — | ⬜ Not started |
| `08` | **Paper Replicating** | Replicating a research paper (Vision Transformer) from scratch | — | ⬜ Not started |
| `09` | **Model Deployment** | Deploying a trained model as a demo app | — | ⬜ Not started |

## What I Actually Learned


## Repo Structure

```
pytorch_zerotohero/
├── 00-pytorch-fundamentals/
│   ├── tensors.ipynb          # Tensor fundamentals
│   └── exercise.ipynb         # Chapter 00 exercises
├── 01-pytorch-workflow/
│   └── workflow.ipynb         # End-to-end PyTorch workflow
├── 02-pytorch-classification/
│   └── classification.ipynb   # Binary & multi-class classification
├── extra/
│   └── cnn_pytorch.py         # CIFAR-10 CNN training: basic CNN vs. small ResNet
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```
