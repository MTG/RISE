# RISE - Raga Independent Svara Encoder

Self-supervised learning for raga independent svara representation, primarily aimed at Carnatic music transcription and related tasks such as performance analysis and melodic pattern recognition. This codebase accompanies the DLFM 2026 submission:

`A Raga Independent Encoder for Svara Representation in Carnatic Music - Vivek Vijayan, Thomas Nuttall, Xavier Serra`

---

## Setup

```bash
git clone https://github.com/MTG/RISE.git
cd RISE
pip install -e .
```

---

## Run experiments

1. Prepare dataset:

```bash
./run.sh preprocess
```

> We extract pitch contours from both the Carnatic Music Rhythm (CMR) and Carnatic Varnam datasets. For the CMR dataset, we sample plausible svara candidates from the pitch contours using the beat annotations, while for the Carnatic Varnam dataset, we use the provided svara annotations.

2. Pre-train the model on [Carnatic Music Rhythm (CMR)](https://zenodo.org/records/1264394) dataset:

```bash
./run.sh pretrain
```

> We pretrain an InceptionTime encoder using the InfoNCE loss on unannotated pitch contours from the CMR dataset. Positive pairs are created by applying data augmentations such as time warping and pitch drifting.

<img src="images/simclr.png" alt="simclr" width="500">

3. Fine-tune the pretrained model on annotated [Carnatic Varnam](https://doi.org/10.5281/zenodo.1257117) dataset using LoRA and report F1 score:

```bash
./run.sh classification
```

> We finetune the pretrained model on annotated data using cross-entropy loss for svara classification. Low-rank adaptation (LoRA) is used for efficient fine-tuning. we report F1 score for baseline and fine-tuned models.

<img src="images/lora.png" alt="simclr" width="500">

4. Cluster svara embeddings on the Carnatic Varnam dataset using HDBSCAN and report Normalized Mutual Information (NMI):

```bash
./run.sh clustering
```

> We evaluate the learned representations by clustering svara embeddings to identify distinct svara forms (gamaka realizations). HDBSCAN is applied independently for each svara, and the resulting clusters are compared against expert-provided svara-form annotations using the Normalized Mutual Information (NMI) score.

5. Retrieve melodic patterns on the [Indian Art Music Melodic Similarity (IAMMS)](https://doi.org/10.5281/zenodo.1266504) dataset and report MAP, MRR and P@k:

```bash
./run.sh pattern_recognition
```

> We split each phrase into svara-length windows, embed every window with the pretrained encoder, and compare phrases by the cosine similarity of their corresponding windows. Retrieval is scored with Mean Average Precision (MAP), Mean Reciprocal Rank (MRR) and Precision@k.

6. Synthesise svara contours from the learned embeddings and report reconstruction error:

```bash
./run.sh synthesis
```

> We attach a transposed-convolutional decoder to the frozen pretrained encoder and train it to reconstruct the original pitch time series. Reconstruction is scored by DTW distance, periodicity error and pitch position error, measured in cents.
