---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Evaluation
permalink: /docs/en/evaluation
key: docs-evaluation
modify_date: "2020-06-05"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

## Spark NLP Evaluation

This module includes tools to evaluate the accuracy of annotators and visualize the parameters used on training. It includes specific metrics for each **annotator** and its training time.
The results will display on the console or to an [MLflow tracking UI](https://mlflow.org/docs/latest/tracking.html). Just with a simple import you can start using eval module.
- Check how to setup MLflow UI
- See [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/enterprise/eval) on eval folder if you want to check specific running examples.

**Example:**

<div class="tabs-box tabs-new pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp_jsl.eval import *
```

```scala
import com.johnsnowlabs.nlp.eval._
```

</div></div><div class="h3-box" markdown="1">

### Evaluating Norvig Spell Checker

You can evaluate this spell checker either by training an annotator or by using a pretrained model.

- spark: Spark session.
- trainFile: A corpus of documents with correctly spell words.
- testFile: A corpus of documents with misspells words.
- groundTruthFile: The same corpus used on *testFile* but with correctly spell words.

**Train File Example:**
```bash
Any document that you prefer with correctly spell words.
```

**Test File Example:**
```bash
My siter go to Munich.
```

**Ground Truth File Example:**
```bash
My sister goes to Munich.
```

**Example for annotator:**

<div class="tabs-box tabs-new pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
spell = NorvigSweetingApproach() \
        .setInputCols(["token"]) \
        .setOutputCol("checked") \
        .setDictionary(dictionary_file)

norvigSpellEvaluation = NorvigSpellEvaluation(spark, test_file, ground_truth_file)
norvigSpellEvaluation.computeAccuracyAnnotator(train_file, spell)
```

```scala
val spell = new NorvigSweetingApproach()
   .setInputCols(Array("token"))
   .setOutputCol("checked")
   .setDictionary(dictionary_file)

val norvigSpellEvaluation = new NorvigSpellEvaluation(spark, testFile, groundTruthFile)
norvigSpellEvaluation.computeAccuracyAnnotator(trainFile, spell)
```

</div>

**Example for pretrained model:**

<div class="tabs-box tabs-new pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
spell = NorvigSweetingModel.pretrained()

norvigSpellEvaluation = NorvigSpellEvaluation(spark, test_file, ground_truth_file)
norvigSpellEvaluation.computeAccuracyModel(spell)
```

```scala
val spell = NorvigSweetingModel.pretrained()
val norvigSpellEvaluation = new NorvigSpellEvaluation(spark, testFile, groundTruthFile)
norvigSpellEvaluation.computeAccuracyModel(spell)
```

</div></div><div class="h3-box" markdown="1">

### Evaluating Symmetric Spell Checker

You can evaluate this spell checker either by training an annotator or by using a pretrained model.

- spark: Spark session
- trainFile: A corpus of documents with correctly spell words.
- testFile: A corpus of documents with misspells words.
- groundTruthFile: The same corpus used on *testFile* but with correctly spell words.

**Train File Example:**

```bash
Any document that you prefer with correctly spell words.
```

**Test File Example:**

```bash
My siter go to Munich.
```

**Ground Truth File Example:**

```bash
My sister goes to Munich.
```

**Example for annotator:**

<div class="tabs-box tabs-new pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```python
spell = SymmetricDeleteApproach() \
        .setInputCols(["token"]) \
        .setOutputCol("checked") \
        .setDictionary(dictionary_file)

symSpellEvaluation = SymSpellEvaluation(spark, test_file, ground_truth_file)
symSpellEvaluation.computeAccuracyAnnotator(train_file, spell)
```

```scala
val spell = new SymmetricDeleteApproach()
      .setInputCols(Array("token"))
      .setOutputCol("checked")

val symSpellEvaluation = new SymSpellEvaluation(spark, testFile, groundTruthFile)
symSpellEvaluation.computeAccuracyAnnotator(trainFile, spell)
```

</div>

**Example for pretrained model:**

<div class="tabs-box tabs-new pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```python
spell = SymmetricDeleteModel.pretrained()

symSpellEvaluation = NorvigSpellEvaluation(spark, test_file, ground_truth_file)
symSpellEvaluation.computeAccuracyModel(spell)
```

```scala
val spell = SymmetricDeleteModel.pretrained()
val symSpellEvaluation = new SymSpellEvaluation(spark, testFile, groundTruthFile)
symSpellEvaluation.computeAccuracyModel(spell)
```

</div></div><div class="h3-box" markdown="1">

### Evaluating NER DL

You can evaluate NER DL when training an annotator.

- spark: Spark session.
- trainFile: Files with labeled NER entities for training. 
- testFile: Files with labeled NER entities for testing. These files are used to evaluate the model. So, it's used for prediction and the labels as ground truth.
- tagLevel: The granularity of tagging when measuring accuracy on entities. Set "IOB" to include inside and beginning, empty to ignore it. For example
to display accuracy for entity I-PER and B-PER set "IOB" whereas just for entity PER set it as an empty string.

**Example:**

<div class="tabs-box tabs-new pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```python
embeddings = WordEmbeddings() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("embeddings") \
            .setEmbeddingsSource("glove.6B.100d.txt", 100, "TEXT")

ner_approach = NerDLApproach() \
      .setInputCols(["document", "token", "embeddings"]) \
      .setLabelColumn("label") \
      .setOutputCol("ner") \
      .setMaxEpochs(10) \
      .setRandomSeed(0)

nerDLEvaluation = NerDLEvaluation(spark, test_File, tag_level)
nerDLEvaluation.computeAccuracyAnnotator(train_file, ner_approach, embeddings)
```

```scala
val embeddings = new WordEmbeddings()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setEmbeddingsSource("glove.6B.100d.txt", 100, WordEmbeddingsFormat.TEXT)

val nerApproach = new NerDLApproach()
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setLabelColumn("label")
  .setOutputCol("ner")
  .setMaxEpochs(10)
  .setRandomSeed(0)

val nerDLEvaluation = new NerDLEvaluation(spark, testFile, tagLevel)
nerDLEvaluation.computeAccuracyAnnotator(trainFile, nerApproach, embeddings)
```

</div>


**Example for pretrained model:**

<div class="tabs-box tabs-new pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```python
ner_dl = NerDLModel.pretrained()

nerDlEvaluation = NerDLEvaluation(spark, test_File, tag_level)
nerDlEvaluation.computeAccuracyModel(ner_dl)
```

```scala
val nerDl = NerDLModel.pretrained()

val nerDlEvaluation = NerDLEvaluation(spark, testFile, tagLevel)
nerDlEvaluation.computeAccuracyModel(nerDl)
```

</div></div><div class="h3-box" markdown="1">

### Evaluating NER CRF

You can evaluate NER CRF when training an annotator.

- spark: Spark session.
- trainFile: Files with labeled NER entities for training. 
- testFile: Files with labeled NER entities for testing. These files are used to evaluate the model. So, it's used for prediction and the labels as ground truth.
- format: The granularity of tagging when measuring accuracy on entities. Set "IOB" to include inside and beginning, empty to ignore it. For example
to display accuracy for entity I-PER and B-PER set "IOB" whereas just for entity PER set it as an empty string.

**Example:**

<div class="tabs-box tabs-new pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```python
embeddings = WordEmbeddings() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("embeddings") \
            .setEmbeddingsSource("glove.6B.100d.txt", 100, "TEXT")

ner_approach = NerCrfApproach() \
      .setInputCols(["document", "token", "pos", "embeddings"]) \
      .setLabelColumn("label") \
      .setOutputCol("ner") \
      .setMaxEpochs(10) \
      .setRandomSeed(0)

nerCrfEvaluation = NerCrfEvaluation(spark, test_File, tag_level)
nerCrfEvaluation.computeAccuracyAnnotator(train_file, ner_approach, embeddings)
```

```scala
val embeddings = new WordEmbeddings()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setEmbeddingsSource("./glove.6B.100d.txt ", 100, WordEmbeddingsFormat.TEXT)
      .setCaseSensitive(true)

val nerTagger = new NerCrfApproach()
  .setInputCols(Array("sentence", "token","pos", "embeddings"))
  .setLabelColumn("label")
  .setOutputCol("ner")
  .setMaxEpochs(10)

val nerCrfEvaluation = new NerCrfEvaluation(testFile, format)
nerCrfEvaluation.computeAccuracyAnnotator(trainFile, nerTagger, embeddings)
```

</div>

**Example for pretrained model:**

<div class="tabs-box tabs-new pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```python
ner_crf = NerCrfModel.pretrained()

nerCrfEvaluation = NerCrfEvaluation(spark, test_File, tag_level)
nerCrfEvaluation.computeAccuracyModel(ner_crf)
```

```scala
nerCrf = NerCrfModel.pretrained()

nerCrfEvaluation = NerCrfEvaluation(spark, testFile, tagLevel)
nerCrfEvaluation.computeAccuracyModel(nerCrf)
```

</div></div>

### Evaluating POS Tagger

You can evaluate POS either by training an annotator or by using a pretrained model.

- spark: Spark session.
- trainFile: A labeled POS file see and example [here](https://nlp.johnsnowlabs.com/docs/en/annotators#pos-dataset).
- testFile: A CoNLL-U format file.

**Example for annotator:**

<div class="tabs-box tabs-new pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```python
pos_tagger = PerceptronApproach() \
             .setInputCols(["document", "token"]) \
             .setOutputCol("pos") \
             .setNIterations(2)

posEvaluation = POSEvaluation(spark, test_file)
posEvaluation.computeAccuracyAnnotator(train_file, pos_tagger)
```

```scala
val posTagger = new PerceptronApproach()
      .setInputCols(Array("document", "token"))
      .setOutputCol("pos")
      .setNIterations(2)

val posEvaluation = new POSEvaluation(spark, testFile)
posEvaluation.computeAccuracyAnnotator(trainFile, posTagger)
```

</div>
