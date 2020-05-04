---
layout: article
title: Training
permalink: /docs/en/training
key: docs-training
modify_date: "2019-10-23"
use_language_switcher: "Python-Scala"

---

## Training Datasets

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

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.training import POS
train_pos = POS().readDataset(spark, "./src/main/resources/anc-pos-corpus")
```

```scala
import com.johnsnowlabs.nlp.training.POS
val trainPOS = POS().readDataset(spark, "./src/main/resources/anc-pos-corpus")
```

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

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.training import CoNLL
training_conll = CoNLL().readDataset(spark, "./src/main/resources/conll2003/eng.train")
```

```scala
import com.johnsnowlabs.nlp.training.CoNLL
val trainingConll = CoNLL().readDataset(spark, "./src/main/resources/conll2003/eng.train")
```

### Spell Checkers Dataset

In order to train a Norvig or Symmetric Spell Checkers, we need to get corpus data as a spark dataframe. We can read a plain text file and transforms it to a spark dataset.  

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```python
train_corpus = spark.read.text("./sherlockholmes.txt")
                    .withColumnRenamed("value", "text")
```

```scala
val trainCorpus = spark.read.text("./sherlockholmes.txt")
                       .select(trainCorpus.col("value").as("text"))
```

### Vivekn Sentiment Analysis Dataset

To train ViveknSentimentApproach, it is needed to have input columns DOCUMENT and TOKEN, and a String column which is set with `setSentimentCol` stating either `positive` or `negative`


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