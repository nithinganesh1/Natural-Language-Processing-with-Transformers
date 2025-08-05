# Natural-Language-Processing-with-Transformers

## What's Novel About Transformers?

To understand what makes Transformers a breakthrough, we must first understand three core concepts:

1. **The Encoder-Decoder Framework**
2. **Attention Mechanisms**
3. **Transfer Learning**

---

## 1. The Encoder-Decoder Framework

Before Transformers, sequence-based tasks were dominated by recurrent architectures.

### 🔄 Recurrent Neural Networks (RNNs & LSTMs)

**Primary Use**: Ideal for modeling sequential data like text, speech, or time series.

**Core Idea**: RNNs have a feedback loop that allows information (the "hidden state") to persist from one step to the next, giving the network a form of memory.

* **LSTMs (Long Short-Term Memory networks)** are an advanced type of RNN that are better at handling long-term dependencies in data.

---

### 🔄 Sequence-to-Sequence (Seq2Seq) Architecture

This framework is crucial for tasks like machine translation, where an input sequence is converted into a different output sequence. It consists of two main components:

#### 🔍 The Encoder

* **Job**: To read the entire input sequence (e.g., an English sentence) and compress all its information into a single, fixed-size vector.
* **Process**: An RNN reads the input word-by-word (e.g., "Anime", "is", "good!").
* **Output**: A numerical representation called the **last hidden state** or **context vector**, summarizing the entire input.

#### 📝 The Decoder

* **Job**: To take the context vector from the encoder and generate the output sequence word-by-word (e.g., the German translation "Anime", "ist", "gut!").
* **Process**: It's another RNN that uses the context to predict the next word at each step.
![Encoder-Decoder Architecture](https://github.com/nithinganesh1/Natural-Language-Processing-with-Transformers/blob/main/images/encoder-decoder.png)
---

### ⚠️ The Bottleneck Problem

The classic Seq2Seq architecture has a significant weakness: the **information bottleneck**.

* The entire meaning of the input sequence—no matter how long—must be squeezed into the single, fixed-size hidden state vector.
* For long sequences, **early input information is often lost**.
* The decoder only has access to this compressed, and potentially incomplete, summary.

What is Attention?

A mechanism that allows the decoder to access **all of the encoder's hidden states** (one for each input word), not just the last one.
How It Helps

* During decoding, attention lets the model **focus on relevant parts of the input** for each word it generates.
* It removes the dependency on a single, compressed context vector.
* This was a **critical advancement** toward the Transformer architecture.

---


## 2. ✨ Attention Mechanisms

### 🧠 Core Idea

Instead of producing a single hidden state, attention allows the encoder to output a hidden state for **each input token**. The decoder can then access these states and decide which ones are most relevant at each output timestep.

### 🎯 How Attention Works

* The decoder **does not use all encoder states equally**.
* It assigns **attention weights** to each hidden state depending on the relevance to the current output prediction.
* These weights determine how much attention is paid to different input tokens at every step.

> For example: In a translation task from English to French, the model might align "zone" to "Area" even though they appear in different positions. This dynamic weighting creates **nontrivial alignments** between source and target words.

![Attention Weights Visualization](https://github.com/nithinganesh1/Natural-Language-Processing-with-Transformers/blob/main/images/weight_corr.png)

### ⛓ Limitation of RNN-based Attention

Even with attention, **recurrent models are still sequential**:

* They cannot process all input tokens in parallel.
* Training takes longer due to step-by-step processing.

---

## 🚀 A Breakthrough: Self-Attention and Transformers

To address the limitations of sequential computation:

* Transformers remove recurrence **entirely**.
* They rely on **self-attention**, which allows each token to attend to **every other token** in the sequence.

### 🔗 What is Self-Attention?

* A form of attention that operates **within the same layer**.
* Used in both encoder and decoder layers.

### ⚡ Why It Matters

* Enables **parallel computation**.
* Dramatically speeds up training.
* Forms the **core of the Transformer** architecture.

![Self-Attention Architecture](https://github.com/nithinganesh1/Natural-Language-Processing-with-Transformers/blob/main/images/self_both_encoder_and_decoder.png)

---

