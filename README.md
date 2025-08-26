# Text Classification

Concise, high-value points on text classification. Focus: concepts, structure, preprocessing, modeling choices, evaluation, and extensions. Not runnable code — just the structure and why each step matters.

If you want the sample code go here: [2Text\_Classification.ipynb](https://github.com/nithinganesh1/Natural-Language-Processing-with-Transformers/blob/main/2Text_Classification.ipynb)

---


## 1) One-line summary

* Text classification: convert raw text → tokens → model inputs → predicted label; two main strategies: **feature extraction** (freeze LM) and **fine‑tuning** (train LM + head).

---

## 2) Full pipeline (collapsed view)

* **Data** → **EDA** → **Preprocessing & Tokenization** → **Encoding & Batching** → **Modeling (A/B)** → **Training** → **Evaluation** → **Error analysis** → **Optimization & Deployment**

---

## 3) Dataset & EDA (what to check, and why)

* Class counts: detect imbalance → affects metrics and loss choices.
* Text length distribution: choose `max_length` / truncation policy.
* Label quality & duplicates: check for noisy/ambiguous labels.
* Language / domain drift: detect out-of-domain examples; consider filtering.
* Baselines: dummy classifier (most\_frequent) to set floor performance.

---

## 4) Tokenization (types & practical consequences)

* **Character tokenization**: no OOV, but long sequences; rarely used alone.
* **Word tokenization**: intuitive, large vocab, OOV problem.
* **Subword tokenization (BPE / WordPiece / Unigram)**: best tradeoff for transformers — reduces OOV, keeps vocab small.
* **Special tokens**: CLS/SEP/PAD; model expects specific tokens and token ids.
* **Tokenizer pitfalls**: inconsistent lowercase/casing vs model type; different model checkpoints require matching tokenizers.

---

## 5) Preprocessing (advanced, important choices)

* **Minimal cleaning**: foreign characters, control chars, excessive whitespace.
* **Choose domain-aware cleaning**: keep emojis/hashtags if signal; remove URLs/mentions only if noise for task.
* **Normalization**: unicode normalization, accent removal (only if needed).
* **Contractions expansion**: beneficial for small models/datasets; less necessary for pretrained LMs.
* **Deduplication**: remove exact duplicates (could leak across splits).
* **Label hygiene**: check ambiguous samples; relabel or drop if necessary.

---

## 6) Tokenization & Encoding details (practical flags)

* `truncation=True`, set sensible `max_length` after EDA (e.g., 128 for tweets).
* **Dynamic padding** via `DataCollatorWithPadding` for efficient batches.
* Keep `attention_mask`; understand `token_type_ids` for pair tasks.
* Save tokenized dataset to disk (cache) if re‑running experiments.

---

## 7) Feature extraction (path A) — structure & choices

* **Goal**: freeze LM, extract fixed vectors, train lightweight classifier.
* **Steps**:

  1. Load `AutoModel` and set `.eval()` and no grad.
  2. Forward tokenized batches → obtain `last_hidden_state` (B, T, H).
  3. Pool tokens into sentence vector: **masked mean pooling** recommended (robust), or CLS token when supported by model.
  4. Optionally L2‑normalize embeddings.
  5. Train classifier (LogisticRegression, SVM, RandomForest, XGBoost).
* **Why**: cheap, fast, reusable; good baseline when GPU limited.
* **Limitations**: embeddings are not task‑specific — may underperform fine‑tuning.

---

## 8) Pooling strategies (impact matters)

* **CLS token output**: default for many models; cheap.
* **Masked mean pooling**: average tokens with attention mask — more robust across models and lengths.
* **Max pooling**: captures salient features; sensitive to noise.
* **Attention pooling / learned pooling**: better but requires training a small layer.

**Smean pooling idea)**

```python
mask = attention_mask.unsqueeze(-1)
pooled = (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
```

---

## 9) Choosing a downstream classifier (feature path)

* **LogisticRegression**: strong, interpretable baseline; tune regularization.
* **SVM**: good for smaller datasets; scale with kernel choice.
* **Tree ensembles**: robust to feature scaling, slower, may need more data.
* **Neural head**: small MLP on top of embeddings if you want differentiable head but still frozen LM.

---

## 10) Fine‑tuning (path B) — structure & training strategies

* **Core idea**: use `AutoModelForSequenceClassification` → model body + randomly initialized head.
* **Key knobs**:

  * **Learning rate**: often small (2e‑5 to 5e‑5).
  * **Epochs**: 3–5 usually; watch validation.
  * **Weight decay / dropout**: regularization.
  * **Warmup steps**: linear warmup helps stability.
  * **Batch size & gradient accumulation**: trade GPU mem vs steps.
  * **Layer freezing / differential LR**: freeze lower layers or use smaller LR for encoder.
* **Trainer API**: supplies training loop, metrics, checkpoints, and `compute_metrics()` hook.
* **TF Keras**: use `TFAutoModelForSequenceClassification` + `to_tf_dataset()` for similar pipeline.

**Minimal `compute_metrics()` snippet**

```python
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1_macro": f1_score(labels, preds, average="macro")}
```

---

## 11) Evaluation — metrics & practices

* **Primary**: Macro‑F1 (handles imbalance) and Accuracy (easy baseline).
* **Per‑class**: precision/recall/F1 per label to detect weak classes.
* **Confusion matrix**: normalized matrix shows class confusions — crucial diagnostic.
* **Calibration**: check reliability plots if probabilities matter.
* **Runtime metrics**: samples/sec, latency for deployment planning.
* **Statistical checks**: report confidence intervals or multiple seeds if comparing models.

---

## 12) Error analysis (how to slice & what to look for)

* **Sort validation examples by loss** (highest loss are most informative). The book demonstrates retrieving these via forward pass and `cross_entropy` loss.
* **Inspect misclassified examples**: look for ambiguous texts, label noise, or tokenization issues.
* **Cluster confusions**: which classes are confused most (from confusion matrix)?
* **Visualize embeddings** (t-SNE / UMAP) to inspect class separability.
* **Token-level inspection**: look at attention patterns or token contributions.

---

## 13) Handling class imbalance

* **Simple**: class weights in loss (e.g., `weight` in CrossEntropyLoss) or `class_weight` in scikit models.
* **Resampling**: oversample minority (with care), undersample majority.
* **Loss engineering**: focal loss for hard examples, label smoothing to regularize.
* **Evaluation**: prefer macro metrics and per‑class reporting.

---

## 14) Regularization & optimization tricks

* **Weight decay** (\~0.01) and dropout in head.
* **Grad clipping** to stabilize training.
* **Mixed precision** (FP16 / BF16) for memory/speed gains.
* **Grad accumulation** to simulate larger batch sizes.
* **Early stopping** & `load_best_model_at_end`.

---

## 15) Efficiency & production considerations

* **Precompute embeddings** for feature path to speed repeated experiments.
* **Distillation / quantization / pruning** to reduce footprint.
* **ONNX / TorchScript** export for faster inference.
* **Batching & dynamic padding** to increase throughput in serving.
* **Monitor drift** after deployment; plan for re‑training.

---

## 16) Advanced extensions & experiments

* **Domain‑adaptive pretraining** (continue MLM on domain data before fine‑tune).
* **Ensembling** of multiple seeds/checkpoints.
* **Data augmentation** (back‑translation, paraphrasing) for low‑data regimes.
* **Multi‑task learning** or label hierarchies.
* **Calibration & uncertainty** (temperature scaling, MC dropout).
* **Interpretability**: attention inspection, LIME/SHAP on embeddings.

---

## 17) Reproducibility & hygiene

* fix seeds across `random`, `numpy`, and `torch`.
* document `MODEL_CKPT`, tokenizer, `max_length`, batch sizes, and training args.
* save tokenizer + model; version datasets used.
* run multiple seeds where ranking models matters.

---

## 18) Minimal checklist needs to be noted

* [ ] Dataset & splits described (size per split).
* [ ] Tokenizer + `max_length` used.
* [ ] Modeling path (Feature vs Fine‑tune) stated.
* [ ] Training args: epochs, LR, batch size, weight decay.
* [ ] Metrics: accuracy, macro‑F1, per‑class F1.
* [ ] Confusion matrix + short error analysis writeup.
* [ ] Hardware used (GPU/CPU; samples/sec).

---

## 19) Short references to the chapter's useful recipes

* Use masked mean pooling when extracting embeddings from `last_hidden_state`.
* For small datasets or no‑GPU contexts, start with feature extraction and a logistic regression baseline.
* Use Trainer + `compute_metrics` for fast experiments and checkpointing.

---
