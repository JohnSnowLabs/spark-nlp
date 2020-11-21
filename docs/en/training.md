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

<div class="h3-box" markdown="1">

### POS Dataset

In order to train a Part of Speech Tagger annotator, we need to get corpus data as a spark dataframe. There is a component that does this for us: it reads a plain text file and transforms it to a spark dataset.  

**Input File Format:**

```bash
A|DT few|JJ months|NNS ago|RB you|PRP received|VBD a|DT letter|NN
```

**Available parameters are:**

- spark: Spark session
- path(string): Path to file with corpus data for training POS
- delimiter(string): Delimiter of token and postag. Defaults to `|`
- outputPosCol(string): Name of the column with POS values. Defaults to "tags".

**Example:**  

Refer to the [POS](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.training.POS) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.training import POS
train_pos = POS().readDataset(spark, "./src/main/resources/anc-pos-corpus")
```

```scala
import com.johnsnowlabs.nlp.training.POS
val trainPOS = POS().readDataset(spark, "./src/main/resources/anc-pos-corpus")
```

</div></div><div class="h3-box" markdown="1">

### CoNLL Dataset

In order to train a Named Entity Recognition DL annotator, we need to get CoNLL format data as a spark dataframe. There is a component that does this for us: it reads a plain text file and transforms it to a spark dataset.

**Constructor parameters:**

- documentCol: String = "document",
- sentenceCol: String = "sentence",
- tokenCol: String = "token",
- posCol: String = "pos",
- conllLabelIndex: Int = 3,
- conllPosIndex: Int = 1,
- conllTextCol: String = "text",
- labelCol: String = "label",
- explodeSentences: Boolean = false

**Available parameters are:**

- spark: Spark session
- path(string): Path to a [CoNLL 2003 IOB NER file](https://www.clips.uantwerpen.be/conll2003/ner).
- readAs(string): Can be LINE_BY_LINE or SPARK_DATASET, with options if latter is used (default LINE_BY_LINE)

**Example:**

Refer to the [CoNLL](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.training.CoNLL) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.training import CoNLL
training_conll = CoNLL().readDataset(spark, "./src/main/resources/conll2003/eng.train")
```

```scala
import com.johnsnowlabs.nlp.training.CoNLL
val trainingConll = CoNLL().readDataset(spark, "./src/main/resources/conll2003/eng.train")
```

</div></div><div class="h3-box" markdown="1">

### Spell Checkers Dataset

In order to train a Norvig or Symmetric Spell Checkers, we need to get corpus data as a spark dataframe. We can read a plain text file and transforms it to a spark dataset.  

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
train_corpus = spark.read.text("./sherlockholmes.txt")
                    .withColumnRenamed("value", "text")
```

```scala
val trainCorpus = spark.read.text("./sherlockholmes.txt")
                       .select(trainCorpus.col("value").as("text"))
```

</div></div><div class="h3-box" markdown="1">

### Vivekn Sentiment Analysis Dataset

To train ViveknSentimentApproach, it is needed to have input columns DOCUMENT and TOKEN, and a String column which is set with `setSentimentCol` stating either `positive` or `negative`

</div>

### PubTator Dataset

The PubTator format includes medical papers' titles, abstracts, and tagged chunks (see [PubTator Docs](http://bioportal.bioontology.org/ontologies/EDAM?p=classes&conceptid=format_3783) and [MedMentions Docs](http://github.com/chanzuckerberg/MedMentions) for more information). We can create a Spark DataFrame from a PubTator text file.

**Available parameters are:**

- spark: Spark session
- path(string): Path to a PubTator File

**Example:**

```scala
import com.johnsnowlabs.nlp.training.PubTator
val trainingPubTatorDF = PubTator.readDataset(spark, "./src/test/resources/corpus_pubtator.txt")
```

### TensorFlow Graphs
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
