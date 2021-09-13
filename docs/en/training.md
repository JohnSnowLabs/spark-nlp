---
layout: docs
header: true
title: Training
permalink: /docs/en/training
key: docs-training
modify_date: "2019-10-23"
use_language_switcher: "Python-Scala"

---

## Training Datasets
These are classes to load common datasets to train annotators for tasks such as
part-of-speech tagging, named entity recognition, spell checking and more.

{% include_relative training_entries/pos.md %}
{% include_relative training_entries/conll.md %}
{% include_relative training_entries/conllu.md %}
{% include_relative training_entries/pubtator.md %}

<div class="h3-box" markdown="1">

### Spell Checkers Dataset (Corpus)

In order to train a Norvig or Symmetric Spell Checkers, we need to get corpus data as a spark dataframe. We can read a plain text file and transforms it to a spark dataset.

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
train_corpus = spark.read \
    .text("./sherlockholmes.txt") \
    .withColumnRenamed("value", "text")
```

```scala
val trainCorpus = spark.read
    .text("./sherlockholmes.txt")
    .select(trainCorpus.col("value").as("text"))
```

</div></div>

## Text Processing
These are annotators that can be trained to process text for tasks such as
dependency parsing, lemmatisation, part-of-speech tagging, sentence detection
and word segmentation.

{% include_relative training_entries/DependencyParser.md %}
{% include_relative training_entries/Lemmatizer.md %}
{% include_relative training_entries/Perceptron.md %}
{% include_relative training_entries/SentenceDetectorDL.md %}
{% include_relative training_entries/TypedDependencyParser.md %}
{% include_relative training_entries/WordSegmenter.md %}

## Spell Checkers
These are annotators that can be trained to correct text.

{% include_relative training_entries/ContextSpellChecker.md %}
{% include_relative training_entries/NorvigSweeting.md %}
{% include_relative training_entries/SymmetricDelete.md %}

## Token Classification
These are annotators that can be trained to recognize named entities in text.

{% include_relative training_entries/NerCrf.md %}
{% include_relative training_entries/NerDL.md %}

## Text Classification
These are annotators that can be trained to classify text into different
classes, such as sentiment.

{% include_relative training_entries/ClassifierDL.md %}
{% include_relative training_entries/MultiClassifierDL.md %}
{% include_relative training_entries/SentimentDL.md %}
{% include_relative training_entries/ViveknSentiment.md %}

## External Trainable Models
These are annotators that are trained in an external library, which are then
loaded into Spark NLP.

{% include_relative training_entries/BertForTokenClassification.md %}
{% include_relative training_entries/DistilBertForTokenClassification.md %}


## TensorFlow Graphs
NER DL uses Char CNNs - BiLSTM - CRF Neural Network architecture. Spark NLP defines this architecture through a Tensorflow graph, which requires the following parameters:

- Tags
- Embeddings Dimension
- Number of Chars

Spark NLP infers these values from the training dataset used in [NerDLApproach annotator](annotators.md#ner-dl) and tries to load the graph embedded on spark-nlp package.
Currently, Spark NLP has graphs for the most common combination of tags, embeddings, and number of chars values:

{:.table-model-big.w7}
| Tags | Embeddings Dimension |
| :--- | :------------------: |
|  10  |       100            |
|  10  |       200            |
|  10  |       300            |
|  10  |       768            |
|  10  |       1024           |
|  25  |       300            |

All of these graphs use an LSTM of size 128 and number of chars 100

In case, your train dataset has a different number of tags, embeddings dimension, number of chars and LSTM size combinations shown in the table above, `NerDLApproach` will raise an **IllegalArgumentException** exception during runtime with the message below:

*Graph [parameter] should be [value]: Could not find a suitable tensorflow graph for embeddings dim: [value] tags: [value] nChars: [value]. Check https://nlp.johnsnowlabs.com/docs/en/graph for instructions to generate the required graph.*

To overcome this exception message we have to follow these steps:

1. Clone [spark-nlp github repo](https://github.com/JohnSnowLabs/spark-nlp)

2. Run python file `create_models` with number of tags, embeddings dimension and number of char values mentioned on your exception message error.

    ```bash
    cd spark-nlp/python/tensorflow
    export PYTHONPATH=lib/ner
    python create_models.py [number_of_tags] [embeddings_dimension] [number_of_chars] [output_path]
    ```

3. This will generate a graph on the directory defined on `output_path argument.

4. Retry training with `NerDLApproach` annotator but this time use the parameter `setGraphFolder` with the path of your graph.

**Note:**  Make sure that you have Python 3 and Tensorflow 1.15.0 installed on your system since `create_models` requires those versions to generate the graph successfully.
**Note:**  We also have a notebook in the same directory if you prefer Jupyter notebook to cerate your custom graph.
