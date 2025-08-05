# Text Classification Notes

Text classification is a fundamental task in Natural Language Processing (NLP) where the goal is to assign predefined labels to text. It's widely used for tasks such as:

* Categorizing customer feedback
* Routing support tickets
* Detecting spam emails

Spam filters in email systems are common examples of text classification in action.

## Hugging Face Ecosystem Libraries

This task introduces three powerful libraries from the Hugging Face ecosystem:

* **Datasets**: For loading and managing datasets.
* **Tokenizers**: For converting text into tokens that models can understand.
* **Transformers**: For using and fine-tuning pre-trained transformer models like BERT or DistilBERT.

Together, these tools allow rapid development from raw data to a fine-tuned model for real-world applications.

---

## Loading a Dataset

Datasets can be loaded in different formats using `load_dataset()`:

| Format | Keyword | Example                                            |
| ------ | ------- | -------------------------------------------------- |
| CSV    | `csv`   | `load_dataset("csv", data_files="my_file.csv")`    |
| Text   | `text`  | `load_dataset("text", data_files="my_file.txt")`   |
| JSONL  | `json`  | `load_dataset("json", data_files="my_file.jsonl")` |

To load the **emotion dataset**:

```python
from datasets import load_dataset
emotions = load_dataset("emotion")
```

Dataset structure:

```python
DatasetDict({
  train: Dataset({ features: ['text', 'label'], num_rows: 16000 }),
  validation: Dataset({ features: ['text', 'label'], num_rows: 2000 }),
  test: Dataset({ features: ['text', 'label'], num_rows: 2000 })
})
```

---

## Checking Class Imbalance

Use `matplotlib` to check class distribution:

```python
import matplotlib.pyplot as plt
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()
```

---

## Words Per Tweet

Measure tweet length:

```python
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()
```

---

## From Text to Tokens

Transformer models like DistilBERT **cannot take raw text** as input. The text must be **tokenized** and **encoded** into numerical vectors.

### Tokenization

Tokenization breaks down text into smaller units called **tokens**. Two basic types:

* **Character Tokenization**: Splits text into individual characters.
* **Word Tokenization**: Splits text by words.

Modern tokenizers (like the one in DistilBERT) use **subword tokenization**, which is a compromise between character and word tokenization. The tokenizer is trained on a corpus to find optimal subunits.

### Converting to IDs (idx)

After tokenization, each token is converted to a corresponding **index (idx)** in the vocabulary. These indices are the numerical inputs for the model.

---

