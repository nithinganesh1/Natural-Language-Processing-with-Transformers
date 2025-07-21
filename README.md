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
img[https://github.com/nithinganesh1/Natural-Language-Processing-with-Transformers/blob/main/images/encoder-decoder.png]
---

### ⚠️ The Bottleneck Problem

The classic Seq2Seq architecture has a significant weakness: the **information bottleneck**.

* The entire meaning of the input sequence—no matter how long—must be squeezed into the single, fixed-size hidden state vector.
* For long sequences, **early input information is often lost**.
* The decoder only has access to this compressed, and potentially incomplete, summary.

---

## 2. ✨ The Solution: Attention

### 📈 What is Attention?

A mechanism that allows the decoder to access **all of the encoder's hidden states** (one for each input word), not just the last one.

### ✅ How It Helps

* During decoding, attention lets the model **focus on relevant parts of the input** for each word it generates.
* It removes the dependency on a single, compressed context vector.
* This was a **critical advancement** toward the Transformer architecture.

---

Stay tuned for the next section: **Understanding Attention in Depth** →
