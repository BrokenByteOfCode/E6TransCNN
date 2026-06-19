### E6TransCNN: Technical Architecture Specification

The **E6TransCNN** is a hybrid CNN-Transformer architecture designed for high-resolution, multi-label classification and self-supervised reconstruction of digital and stylized art. 

The architecture is trained in two distinct phases:
1. **Self-Supervised Pre-training** using a customized Masked Autoencoder (MAE) framework optimized for spatial feature reconstruction.
2. **Supervised Multi-Head Fine-tuning** for domain-specific tag classification (Species, General, Meta, and Rating).

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
                     |                     +-- Head 1: General (Sigmoid/BCE)
           [ Transformer Decoder ]         +-- Head 2: Species (Sigmoid/Metric)
                     |                     +-- Head 3: Rating (Softmax/CE)
          [ Pixel Reconstruction ]         +-- Head 4: Meta (Sigmoid/BCE)
```

---

### 1. Architectural Components

#### 1.1 CNN Stem (Local Feature Extractor)
To preserve fine spatial structures (such as line art and outlines) without the high computational cost of running a Transformer on a raw $816 \times 816$ grid, the input is first processed by a lightweight convolutional stem.

* **Input Dimensions:** $816 \times 816 \times 3$
* **Operations:** Three blocks of strided convolutions with **Mish** activation and Group Normalization (stable at low batch sizes).
* **Reduction Factor:** $16 \times$ spatial reduction.
* **Output Dimensions:** $51 \times 51 \times 512$ (representing $2,601$ spatial feature tokens).

Using a convolutional stem before the Transformer (similar to the ConvMAE approach) ensures that early-stage local geometries, such as edges and fills, are captured before spatial tokenization.

---

#### 1.2 Masked Tokenization & Projection (MAE Stage)
During the self-supervised pre-training phase, the spatial feature map ($51 \times 51 \times 512$) is flattened into a sequence of $2,601$ tokens, each representing a $16 \times 16$ pixel area of the original image.

* **Masking Ratio:** $75\%$ (consistent with standard MAE formulations).
* **Process:** 
  1. A random $75\%$ of the spatial tokens are discarded.
  2. The remaining $25\%$ ($650$ tokens) are kept.
  3. Sinusoidal 2D Positional Embeddings are added to the unmasked tokens to retain coordinate information.
  4. These tokens are projected to the Transformer dimension ($d_{model} = 768$).

---

#### 1.3 Transformer Encoder (Global Context)
The $650$ unmasked tokens are processed by a standard Vision Transformer (ViT) encoder.

* **Depth:** 12 layers.
* **Attention Heads:** 12.
* **Dimension ($d_{model}$):** 768.
* **Feed-Forward Dimension ($d_{ff}$):** 3072.
* **Mechanism:** Multi-Head Self-Attention (MHSA). This allows the model to learn global spatial relationships (e.g., body proportions, composition, and interactions) from the sparse unmasked tokens.

---

#### 1.4 MAE Decoder (Reconstruction Network)
The decoder is only active during the self-supervised pre-training phase.

* **Input:** The encoded tokens ($650$ tokens) are merged with learnable, shared **[mask]** tokens representing the missing $75\%$. Positional embeddings are re-applied to the full set of $2,601$ tokens.
* **Depth:** 4 layers (shallow decoder to force the representation learning into the encoder).
* **Dimension:** 512.
* **Output Projection:** A linear layer projects the output back to the original pixel space ($16 \times 16 \times 3 = 768$ values per patch) to reconstruct the $816 \times 816 \times 3$ image.

---

#### 1.5 Multi-Head Classifier (Downstream Inference)
For the supervised classification phase, the MAE decoder is removed. The output of the Transformer Encoder is globally pooled (Average Pooling) to produce a 768-dimensional latent vector, which is routed into four independent prediction heads:

1. **Rating Head:** 
   * **Structure:** Linear layer $\to$ Logits $\to$ Softmax.
   * **Output:** 3 classes (Safe, Questionable, Explicit).
2. **Species Head:** 
   * **Structure:** Metric-based Cosine Similarity head (to map features directly to species prototype vectors).
   * **Output:** $N_{species}$ classes. Designed for few-shot performance on rare classes.
3. **General Head:** 
   * **Structure:** Linear layer $\to$ Sigmoid.
   * **Output:** $N_{general}$ classes (multi-label tags representing actions, attributes, and environment).
4. **Meta Head:** 
   * **Structure:** Linear layer $\to$ Sigmoid.
   * **Output:** $N_{meta}$ classes (technical attributes like resolution, medium, and artist signature indicators).

---

### 2. Training Phases & Optimization

#### Phase 1: Self-Supervised Pre-training (Unsupervised)
* **Goal:** Learn domain-specific visual representations without labels.
* **Loss Function:** Mean Squared Error (MSE) calculated **only** on the masked patches in the pixel space:
  $$\mathcal{L}_{rec} = \frac{1}{|M|} \sum_{i \in M} \|x_i - \hat{x}_i\|^2$$
  *(Where $M$ is the set of masked patch indices, $x$ is the original patch, and $\hat{x}$ is the reconstructed patch).*

#### Phase 2: Supervised Fine-tuning (Multi-Task)
* **Goal:** Map the learned latent representations to structured tags.
* **Joint Loss Function:**
  $$\mathcal{L}_{total} = \alpha \mathcal{L}_{rating} + \beta \mathcal{L}_{species} + \gamma \mathcal{L}_{general} + \delta \mathcal{L}_{meta}$$
  * $\mathcal{L}_{rating}$: Cross-Entropy Loss.
  * $\mathcal{L}_{species}$ / $\mathcal{L}_{general}$ / $\mathcal{L}_{meta}$: Asymmetric Loss (ASL) or weighted Binary Cross Entropy (BCE) to handle sparse positive labels and the long-tail class imbalance.

---

### 3. Implementation and Scaling Specifications

* **Target Resolution:** $816 \times 816$ pixels, maintaining native aspect ratios via letterbox padding.
* **Patch Layout:** $51 \times 51$ grid, $16 \times 16$ pixel patches.
* **Pre-processing:** Lanczos interpolation for downscaling to minimize aliasing artifacts in line art.
* **Memory Optimizations:** Gradient checkpointing enabled during pre-training to fit the Transformer sequence length on memory-constrained GPUs. Use of FP16 mixed-precision training.