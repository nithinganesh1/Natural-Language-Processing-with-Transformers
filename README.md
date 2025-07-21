# Natural-Language-Processing-with-Transformers

Core Concepts Behind Transformers
To understand what makes Transformers novel, we need to grasp three key ideas:

The Encoder-Decoder Framework

Attention Mechanisms

Transfer Learning

The Encoder-Decoder Framework
Before Transformers, Recurrent Neural Networks (RNNs), especially LSTMs (Long Short-Term Memory networks), were state-of-the-art for sequence modeling.

RNNs/LSTMs
Function: Designed for sequential data like text. They have a feedback loop that allows information from one step to pass to the next.

Process: An RNN processes an input sequence (e.g., a sentence) one word at a time. At each step, it updates a hidden state, which acts as its memory of what it has seen so far.

Sequence-to-Sequence (Seq2Seq) Architecture
This architecture is used for tasks like machine translation, where an input sequence is transformed into a different output sequence. It has two main parts:

Encoder:

An RNN that reads the entire input sentence, word by word (e.g., "Anime", "is", "good!").

Its goal is to compress the meaning of the whole sentence into a single vector, called the last hidden state or context vector.

Decoder:

Another RNN that takes the encoder's final hidden state.

It uses this context to generate the output sequence one word at a time (e.g., "Anime", "ist", "gut!").

The Bottleneck Problem 🤔
The simple encoder-decoder model has a major weakness: the information bottleneck. The single hidden state vector passed from the encoder to the decoder has to represent the entire meaning of the input sequence. For long sentences, it's very difficult to cram all the necessary information into this one fixed-size vector, and details from the beginning of the sentence are often lost.

Introducing Attention ✨
Attention is the mechanism created to solve this bottleneck problem.

What it is: Instead of just using the encoder's final hidden state, attention allows the decoder to look back at all the hidden states from the encoder at every step of the generation process.

Why it helps: This allows the decoder to focus on the most relevant parts of the input sentence when generating a specific word in the output, overcoming the memory limitations of the older architecture.
