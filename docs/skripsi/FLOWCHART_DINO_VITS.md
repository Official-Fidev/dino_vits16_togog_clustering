# Flowchart DINO ViT-S16 (Self-Supervised Feature Extractor)

Dokumen ini menyajikan diagram alur kerja internal arsitektur **DINO ViT-S16** sebagai tulang punggung ekstraksi fitur dalam sistem *unsupervised clustering* citra patung.

---

## 1. Alur Utama DINO ViT-S16

Gambaran besar bagaimana satu citra diproses dari piksel mentah menjadi vektor fitur 768-dimensi.

```mermaid
graph TD
    A["Citra Input\n224×224"] --> B["Patch Embedding\n196 Patch × 16×16 px"]
    B --> C["[CLS] Token + Positional Encoding"]
    C --> D["Transformer Encoder\n12 Layer Self-Attention"]
    D --> E["[CLS] Token Output"]
    E --> F["Vektor Fitur 768-Dimensi\n→ features.npy"]
```

---

## 2. Detail Mekanisme Patch Embedding

Proses pemecahan citra menjadi *patch* sebelum dimasukkan ke dalam *Transformer Encoder*.

```mermaid
graph LR
    A["Citra Input\n224×224 piksel"] --> B["Bagi menjadi Grid\n14×14 = 196 Patch"]
    B --> C["Tiap Patch: 16×16×3\n= 768 nilai piksel"]
    C --> D["Linear Projection\n768 → 384 dimensi"]
    D --> E["196 Patch Embeddings"]
    E --> F["Gabungkan dengan\n[CLS] Token Learnable"]
    F --> G["197 Token Sequence\n(1 CLS + 196 Patch)"]
```

---

## 3. Detail Mekanisme Multi-Head Self-Attention

Bagaimana setiap *Transformer Layer* memproses relasi antar token melalui mekanisme *self-attention*.

```mermaid
graph TD
    In["Input Token Sequence\n(197 × 384)"] --> QKV["Linear Projection\nHitung Q, K, V per Head"]

    subgraph Heads ["6 Attention Heads (Paralel)"]
        QKV --> H1["Head 1\nQ·Kᵀ / √dk"]
        QKV --> H2["Head 2\nQ·Kᵀ / √dk"]
        QKV --> H3["Head 3 ... 6\nQ·Kᵀ / √dk"]
    end

    H1 --> Soft1["Softmax\n(Attention Weights)"]
    H2 --> Soft2["Softmax\n(Attention Weights)"]
    H3 --> Soft3["Softmax\n(Attention Weights)"]

    Soft1 --> AV1["× Value (V)"]
    Soft2 --> AV2["× Value (V)"]
    Soft3 --> AV3["× Value (V)"]

    AV1 --> Concat["Concat Semua Head\nOutput"]
    AV2 --> Concat
    AV3 --> Concat

    Concat --> Proj["Linear Projection\nOutput"]
    Proj --> Res["Residual Connection\n+ Layer Normalization"]
    Res --> FFN["Feed-Forward Network\n(2 Linear Layer + GELU)"]
    FFN --> Out["Output Token Sequence\n(197 × 384)"]
```

---

## 4. DINO Self-Supervised Training Objective (Knowledge Distillation)

Bagaimana DINO melatih model tanpa label melalui mekanisme *student–teacher distillation*.

```mermaid
graph TD
    Img["Citra Asli"] --> Aug1["Augmentasi Global\n(Crop Besar ≥ 50%)"]
    Img --> Aug2["Augmentasi Lokal\n(Crop Kecil < 50%)"]

    Aug1 --> Teacher["Teacher Network\n(Parameter: EMA dari Student)"]
    Aug1 --> Student["Student Network\n(Parameter: Diperbarui via Backprop)"]
    Aug2 --> Student

    Teacher --> TOut["Output Softmax Teacher\n(Stop-Gradient)"]
    Student --> SOut["Output Softmax Student"]

    TOut --> Loss["Cross-Entropy Loss\n(Student ≈ Teacher)"]
    SOut --> Loss

    Loss --> BP["Backpropagation\nUpdate Student Weights"]
    BP --> EMA["Exponential Moving Average\nUpdate Teacher Weights"]
    EMA --> Teacher
```

---

## 5. Alur Inference: Ekstraksi Fitur Batch

Alur operasional saat DINO ViT-S16 digunakan untuk mengekstraksi fitur dari seluruh dataset (mode *inference*, tanpa pelatihan).

```mermaid
graph TD
    Start["Mulai: Muat Dataset\n(manifest.csv + Tensor .pt)"] --> Dev{"Deteksi\nPerangkat Keras"}

    Dev --> |"CUDA Tersedia"| GPU["Alokasi Model ke GPU"]
    Dev --> |"Hanya CPU"| CPU["Alokasi Model ke CPU"]

    GPU --> Batch["Buat DataLoader\nBatch Size: 32"]
    CPU --> Batch

    Batch --> Loop{"Iterasi Batch"}

    Loop --> |"Tiap Batch"| NoGrad["torch.no_grad()\n(Matikan Gradient)"]
    NoGrad --> Forward["Forward Pass\nvia Transformer Encoder"]
    Forward --> CLS["Ambil [CLS] Token\n(Index ke-0)"]
    CLS --> Collect["Kumpulkan ke Buffer"]
    Collect --> Loop

    Loop --> |"Selesai"| Stack["Stack Semua Vektor\n(N × 768 Matrix)"]
    Stack --> Save["Simpan features.npy"]
    Save --> End["Selesai: Fitur Siap\nuntuk UMAP & HDBSCAN"]
```

---

> **Catatan:** Model DINO ViT-S16 yang digunakan adalah *pre-trained* dari Facebook Research dan **tidak** dilatih ulang (*frozen weights*). Seluruh tahap pelatihan DINO (Knowledge Distillation) dijelaskan di Diagram 4 sebagai konteks teoritis arsitektur, bukan sebagai bagian dari pipeline eksperimen.
