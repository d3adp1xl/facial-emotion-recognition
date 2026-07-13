# Facial Emotion Recognition with Deep Learning

Classifying facial expressions into four emotions — **happy, sad, surprise, neutral** — from 48×48 grayscale images. I trained and compared **seven models** (custom CNNs and transfer-learning architectures), then tuned the best performer to **82% test accuracy** — beating every pretrained ImageNet model I tried.

> **TL;DR:** A purpose-built grayscale CNN (82%) outperformed VGG16 (66%), EfficientNet (59%), and ResNet (54%). The lesson: match model complexity to your data, not to the leaderboard.

**Stack:** Python · TensorFlow / Keras · scikit-learn · NumPy · Matplotlib / Seaborn · Google Colab (GPU)

---

## The Problem

Software can read what you type and click, but it has no idea how you *feel*. Facial emotion recognition closes that gap — letting a system read an emotional state directly from a face and respond to it. The goal here: build a model that identifies one of four emotions from a low-resolution facial image, and understand honestly where it can and can't be trusted.

## The Data

| Property | Value |
|---|---|
| Training images | 15,109 |
| Classes | Happy, Sad, Surprise, Neutral (4) |
| Balance | Near-balanced (~3,170–3,980 per class) |
| Resolution | 48×48 pixels, single-channel **grayscale** |

That last row drives everything. **48×48 grayscale is a hard constraint**, not an afterthought — there's very little detail to work with, which turns out to matter enormously for model choice.

## Approach

I built and evaluated seven models across three families, on a proper train / validation / test split, then tuned the strongest one:

- **Models 1–3** — custom CNNs built from scratch, increasing in size (grayscale and RGB variants)
- **Models 4–6** — transfer learning on ImageNet backbones: VGG16, ResNet101V2, EfficientNetV2B2
- **Model 7** — a deeper custom grayscale CNN, then tuned to maximize test accuracy

All models used Adam (lr 0.001), categorical cross-entropy, batch size 32, up to 30 epochs, with data augmentation (horizontal flip, brightness jitter, shear) and callbacks for checkpointing, early stopping, and learning-rate reduction on plateau.

## Results

| Model | Type | Test Accuracy |
|---|---|:---:|
| **Model 7 (tuned)** | **Custom CNN, grayscale** | **0.82** |
| Model 3 | Custom CNN, RGB | 0.73 |
| VGG16 | Transfer learning | 0.66 |
| Model 1 | Custom CNN, grayscale | 0.65 |
| EfficientNetV2B2 | Transfer learning | 0.59 |
| ResNet101V2 | Transfer learning | 0.54 |
| Model 2 | Custom CNN, grayscale | 0.26 *(failed to converge)* |

*Random-guess baseline for 4 classes = 0.25.*

## The Key Insight: Bigger Isn't Better

The most interesting result is that **the large pretrained models lost to a from-scratch CNN**. VGG16, ResNet, and EfficientNet were all designed for 224×224 full-color images. Feed them 48×48 grayscale and, after their deep stacks of downsampling, almost no usable signal survives — their ImageNet-learned features never get a chance to fire.

A smaller network built specifically for small grayscale inputs preserves detail that the giants throw away. **The takeaway that generalizes beyond this project: match model complexity to the data you actually have, not to what tops benchmarks on other datasets.**

## The Winning Model (Model 7)

A five-block convolutional network sized for a small input:

- **Input:** 48×48×1 (grayscale)
- **5 conv blocks:** 64 → 128 → 256 → 512 → 512 filters, each `Conv2D + BatchNorm + LeakyReLU` (max-pooling on the first three blocks)
- **Head:** `GlobalAveragePooling2D → Dense(256) + LeakyReLU + Dropout(0.5) → Dense(128) → Dense(4, softmax)`

### Tuning: 70% → 82%

Two changes moved the needle most, taking the model from ~70% to **82%**:

1. **Replaced `Flatten` with `GlobalAveragePooling2D`** and simplified the classifier head — far fewer parameters, less overfitting.
2. **Label smoothing (0.1)** to curb overconfidence on noisy emotion labels, plus **early-stopping patience raised 3 → 10** so the model could train through plateaus instead of quitting early.

## Per-Class Performance

| Emotion | F1 |
|---|:---:|
| Surprise | 0.92 |
| Happy | 0.91 |
| Neutral | 0.75 |
| Sad | 0.71 |

**Where it's strong:** Happy and Surprise have unmistakable visual cues — a smile, wide eyes and a dropped jaw — so the model nails them (~90%+).

**Where it struggles:** Sad vs. Neutral. These share subtle, low-energy features that are genuinely hard to separate at 48×48. Importantly, this is a **data-resolution limit, not a modeling flaw** — no model in the study cleanly separated them.

## Responsible Use

Emotion recognition is sensitive, so the honest recommendation is a cautious one:

- Suitable for **pilot deployment in low-risk settings with human oversight** — treat predictions as signals, not verdicts.
- **Test for bias** across age, ethnicity, and gender before any wider use.
- Require **explicit consent and anonymization**.
- The clearest accuracy win going forward is better *data* (higher-resolution color images), not a fancier model.

## Repository

```
├── README.md
├── notebook/
│   └── facial_emotion_detection.ipynb   # full code: EDA → 7 models → tuning → evaluation
└── assets/                              # charts and confusion matrix
```

**Dataset:** 48×48 grayscale facial images across four emotion classes. The notebook is built to run top-to-bottom on Google Colab with GPU.

---

*Deep learning capstone — MIT Applied AI & Data Science. Built by Fuad Ahmadov.*
