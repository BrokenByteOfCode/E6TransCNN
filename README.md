UPDATE: THIS PROJECT VIOLATES ToS OF E621. FUCK YOU DRAGONFRUIT

# E6TransCNN: Hybrid CNN-Transformer for Digital Art Representation

The **E6TransCNN** is a hybrid CNN-Transformer architecture designed for high-resolution, multi-label classification and self-supervised reconstruction of digital and stylized art. 

The architecture is trained in two distinct phases:
1. **Self-Supervised Pre-training** using a customized Masked Autoencoder (MAE) framework optimized for spatial feature reconstruction.
2. **Supervised Multi-Head Fine-tuning** for domain-specific tag classification (Species, Gender, General, Meta, and Rating).

---

```
                       [ Input Image: 816 x 816 x 3 ]
                                     |
                             [ CNN Stem / Block ]
                                     |
                       [ Feature Map: 51 x 51 x 512 ]
                                     |
                     +---------------+---------------+
                     |                               |
          (Pre-training Phase)             (Fine-tuning Phase)
                     |                               |
            [ 75% Spatial Masking ]        [ Global Avg Pooling ]
                     |                               |
           [ Transformer Encoder ]         [ Multi-Head Classifier ]
                     |                     +-- Head 1: General (Sigmoid/ASL)
           [ Transformer Decoder ]         +-- Head 2: Species (Metric Cosine)
                     |                     +-- Head 3: Rating (Softmax/CE)
          [ Pixel Reconstruction ]         +-- Head 4: Meta (Sigmoid/ASL)
                                           +-- Head 5: Gender (Sigmoid/BCE)
```

---

## Repository Structure

```text
.
├── dataset/
│   ├── collect.py         # Dynamic media scraper (API data harvester)
|   ├── build_vocab.py         # Vocabulary builder and frequency filters
│   └── cleanup.py         # Data sanitization and orphan file sweep tool
├── model/
│   ├── backbone/
│   │   ├── stem.py        # 16x Convolutional reduction block (Mish + GN)
│   │   └── transformer.py # 12-layer ViT Encoder
│   ├── heads/
│   │   ├── classification.py # Rating, Gender, General, Meta projection heads
│   │   └── metric.py      # Species Cosine Metric Learning head
│   ├── losses/
│   │   ├── asymmetric.py  # Asymmetric Loss (ASL) for multi-label tags
│   │   └── dynamic.py     # Kendall multi-task uncertainty loss balancer
│   ├── ssl/
│   │   ├── masking.py     # 2D Sincos Position Embeddings & 75% masking index generator
│   │   └── decoder.py     # 4-layer MAE reconstruction Decoder
│   ├── config.py          # Unified config generator with vocab auto-mapping
│   └── e6trans_cnn.py     # Main model orchestrator
├── usage.py               # Local usage script
└── train.py               # Complete two-phase training loop
```

---

## 1. Architectural Components

### 1.1 CNN Stem (Local Feature Extractor)
To preserve fine spatial structures (such as line art and outlines) without the high computational cost of running a Transformer on a raw $816 \times 816$ grid, the input is first processed by a lightweight convolutional stem.

* **Input Dimensions:** $816 \times 816 \times 3$
* **Operations:** Three blocks of strided convolutions with **Mish** activation and Group Normalization (stable at low batch sizes).
* **Reduction Factor:** $16 \times$ spatial reduction.
* **Output Dimensions:** $51 \times 51 \times 512$ (representing $2,601$ spatial feature tokens).

Using a convolutional stem before the Transformer (similar to the ConvMAE approach) ensures that early-stage local geometries, such as edges and fills, are captured before spatial tokenization.

---

### 1.2 Masked Tokenization & Projection (MAE Stage)
During the self-supervised pre-training phase, the spatial feature map ($51 \times 51 \times 512$) is flattened into a sequence of $2,601$ tokens, each representing a $16 \times 16$ pixel area of the original image.

* **Masking Ratio:** $75\%$ (consistent with standard MAE formulations).
* **Process:** 
  1. A random $75\%$ of the spatial tokens are discarded.
  2. The remaining $25\%$ ($650$ tokens) are kept.
  3. Sinusoidal 2D Positional Embeddings are added to the unmasked tokens to retain coordinate information.
  4. These tokens are projected to the Transformer dimension ($d_{model} = 768$).

---

### 1.3 Transformer Encoder (Global Context)
The $650$ unmasked tokens are processed by a standard Vision Transformer (ViT) encoder.

* **Depth:** 12 layers.
* **Attention Heads:** 12.
* **Dimension ($d_{model}$):** 768.
* **Feed-Forward Dimension ($d_{ff}$):** 3072.
* **Mechanism:** Multi-Head Self-Attention (MHSA). This allows the model to learn global spatial relationships (e.g., body proportions, composition, and interactions) from the sparse unmasked tokens.

---

### 1.4 MAE Decoder (Reconstruction Network)
The decoder is only active during the self-supervised pre-training phase.

* **Input:** The encoded tokens ($650$ tokens) are merged with learnable, shared **[mask]** tokens representing the missing $75\%$. Positional embeddings are re-applied to the full set of $2,601$ tokens.
* **Depth:** 4 layers (shallow decoder to force the representation learning into the encoder).
* **Dimension:** 512.
* **Output Projection:** A linear layer projects the output back to the original pixel space ($16 \times 16 \times 3 = 768$ values per patch) to reconstruct the $816 \times 816 \times 3$ image.

---

### 1.5 Multi-Head Classifier (Downstream Inference)
For the supervised classification phase, the MAE decoder is removed. The output of the Transformer Encoder is globally pooled (Average Pooling) to produce a 768-dimensional latent vector, which is routed into five independent prediction heads:

1. **Rating Head:** 
   * **Structure:** Linear layer to raw logits.
   * **Output:** 3 classes (Safe, Questionable, Explicit).
2. **Gender Head:**
   * **Structure:** Linear layer to raw logits.
   * **Output:** 4 classes (Male, Female, Androgynous, Ambiguous).
3. **Species Head:** 
   * **Structure:** Metric-based Cosine Similarity head (maps features directly to species prototype vectors).
   * **Output:** $N_{species}$ classes. Designed for few-shot performance on rare classes.
4. **General Head:** 
   * **Structure:** Linear layer to raw logits.
   * **Output:** $N_{general}$ classes (multi-label tags representing actions, attributes, and environment).
5. **Meta Head:** 
   * **Structure:** Linear layer to raw logits.
   * **Output:** $N_{meta}$ classes (technical attributes like resolution, medium, and artist signature indicators).

---

## 2. Training Phases & Optimization

### Phase 1: Self-Supervised Pre-training (Unsupervised)
* **Goal:** Learn domain-specific visual representations without labels.
* **Loss Function:** Mean Squared Error (MSE) calculated **only** on the masked patches in the pixel space:
  $$\mathcal{L}_{rec} = \frac{1}{|M|} \sum_{i \in M} \|x_i - \hat{x}_i\|^2$$
  *(Where $M$ is the set of masked patch indices, $x$ is the original patch, and $\hat{x}$ is the reconstructed patch).*

### Phase 2: Supervised Fine-tuning (Multi-Task)
* **Goal:** Map the learned latent representations to structured tags.
* **Joint Loss Function:**
  $$\mathcal{L}_{total} = U_1\mathcal{L}_{rating} + U_2\mathcal{L}_{gender} + U_3\mathcal{L}_{species} + U_4\mathcal{L}_{general} + U_5\mathcal{L}_{meta}$$
  * $\mathcal{L}_{rating}$: Cross-Entropy Loss.
  * $\mathcal{L}_{gender}$: BCE With Logits Loss.
  * $\mathcal{L}_{species}$ / $\mathcal{L}_{general}$ / $\mathcal{L}_{meta}$: Asymmetric Loss (ASL) to handle sparse positive labels and the long-tail class imbalance.
  * $U_{1..5}$: Dynamic weights based on homoscedastic uncertainty optimization.

---

## 3. Deployment & Execution Guide

### Step 1: Environment Setup
Ensure you are using PyTorch 2.x and have a CUDA-enabled GPU. Install required dependencies:
```zsh
pip install torch torchvision pillow requests
```

### Step 2: Data Collection
Harvest digital art data with automatically calculated dynamic balancing. Heavy animations, videos, and flash files (.gif, .webm, .mp4, .swf) are skipped dynamically during the API parse stage:
```zsh
python dataset/collect.py
```

### Step 3: Data Sanitation (Optional but recommended)
Run the cleanup script to remove any existing corrupt files, non-static media, or orphan data files to keep the storage clean and valid:
```zsh
python dataset/cleanup.py
```

### Step 4: Build Vocabularies
Compile metadata tags into discrete indexes. Set the minimum frequency threshold (`min_freq`) to filter out rare tags and noise:
```zsh
python build_vocab.py --min_freq 5
```

### Step 5: Start Training

#### Run Phase 1: Self-Supervised Learning (SSL/MAE)
Pre-train the backbone using reconstruction. This stage optimizes the CNN Stem and Vision Transformer Encoder to understand general digital art geometry:
```zsh
python train.py --phase ssl --batch_size 8 --epochs 100 --lr 1e-4
```

#### Run Phase 2: Supervised Fine-Tuning
Fine-tune the model with multi-task prediction heads. The model automatically loads the pre-trained weights from `checkpoints/model_best_ssl.pt` to initialize the backbone:
```zsh
python train.py --phase classify --batch_size 8 --epochs 50 --lr 5e-5
```

#### Resume From Sudden Interruptions
If a training session crashes, append the `--resume` flag to pick up exactly from the last saved state (epoch, optimizer state, scheduler step, loss balancer parameters, and best loss markers):
```zsh
python train.py --phase classify --resume
```
