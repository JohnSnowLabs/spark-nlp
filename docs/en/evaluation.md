---
layout: article
title: Evaluation
permalink: /docs/en/evaluation
key: docs-evaluation
modify_date: "2019-08-04"
---

## Spark NLP Evaluation

This module includes tools to evaluate the accuracy of annotators. It includes specific metrics for each **annotator** and its training time.
The results will display on the console or to an [MLflow tracking UI](https://mlflow.org/docs/latest/tracking.html).
Just whit a simple import you can start using it.

**Example:**
{% highlight python %}
from sparknlp.eval import *
{% endhighlight %}

{% highlight scala %}
import com.johnsnowlabs.nlp.eval._
{% endhighlight %}

### Evaluating Norvig Spell Checker

You can evaluate this spell checker either training an annotator or using a pretrained model.

- trainFile: A corpus of documents with correctly spell words.
- testFile: A corpus of documents with misspells words.
- groundTruthFile: The same corpus used on *testFile* but with correctly spell words.

**Example for annotator:**
{% highlight python %}
spell = NorvigSweetingApproach() \
        .setInputCols(["token"]) \
        .setOutputCol("checked") \
        .setDictionary(dictionary_file)

norvigSpellEvaluation = NorvigSpellEvaluation(test_file, ground_truth_file)
norvigSpellEvaluation.computeAccuracyAnnotator(train_file, spell)
{% endhighlight %}

{% highlight scala %}
val spell = new NorvigSweetingApproach()
   .setInputCols(Array("token"))
   .setOutputCol("checked")
   .setDictionary(dictionary_file)

val norvigSpellEvaluation = new NorvigSpellEvaluation(testFile, groundTruthFile)
norvigSpellEvaluation.computeAccuracyAnnotator(trainFile, spell)
{% endhighlight %}

**Example for pretrained model:**
{% highlight python %}
spell = NorvigSweetingModel.pretrained()

norvigSpellEvaluation = NorvigSpellEvaluation(test_file, ground_truth_file)
norvigSpellEvaluation.computeAccuracyModel(spell)
{% endhighlight %}

{% highlight scala %}
val spell = NorvigSweetingModel.pretrained()
val norvigSpellEvaluation = new NorvigSpellEvaluation(testFile, groundTruthFile)
norvigSpellEvaluation.computeAccuracyModel(spell)
{% endhighlight %}

### Evaluating Symmetric Spell Checker

You can evaluate this spell checker either training an annotator or using a pretrained model.

- trainFile: A corpus of documents with correctly spell words.
- testFile: A corpus of documents with misspells words.
- groundTruthFile: The same corpus used on *testFile* but with correctly spell words.

**Example for annotator:**
{% highlight python %}
spell = SymmetricDeleteApproach() \
        .setInputCols(["token"]) \
        .setOutputCol("checked") \
        .setDictionary(dictionary_file)

symSpellEvaluation = SymSpellEvaluation(test_file, ground_truth_file)
symSpellEvaluation.computeAccuracyAnnotator(train_file, spell)
{% endhighlight %}

{% highlight scala %}
val spell = new SymmetricDeleteApproach()
      .setInputCols(Array("token"))
      .setOutputCol("checked")

val symSpellEvaluation = new SymSpellEvaluation(testFile, groundTruthFile)
symSpellEvaluation.computeAccuracyAnnotator(trainFile, spell)
{% endhighlight %}

**Example for pretrained model:**
{% highlight python %}
spell = SymmetricDeleteModel.pretrained()

symSpellEvaluation = NorvigSpellEvaluation(test_file, ground_truth_file)
symSpellEvaluation.computeAccuracyModel(spell)
{% endhighlight %}

{% highlight scala %}
val spell = SymmetricDeleteModel.pretrained()
val symSpellEvaluation = new SymSpellEvaluation(testFile, groundTruthFile)
symSpellEvaluation.computeAccuracyModel(spell)
{% endhighlight %}

### Evaluating NER DL

You can evaluate NER DL when training an annotator.

- spark: Spark session
- trainFile: Files with labeled NER entities for training. 
- modelPath: Path to save the model. When the path exists it loads the model instead of training it.
- testFile: Files with labeled NER entities. These files are used to evaluate the model. So, it's used for prediction and the labels as ground truth.
- tagLevel: The granularity of tagging when measuring accuracy on entities. Set "IOB" to include inside and beginning, empty to ignore it. For example
to display accuracy for entity I-PER and B-PER set "IOB" whereas just for entity PER set it as an empty string.

**Example:**
{% highlight python %}
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
{% endhighlight %}

{% highlight scala %}
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
{% endhighlight %}

**Example for pretrained model:**
{% highlight python %}
ner_dl = NerDLModel.pretrained()

nerDlEvaluation = NerDLEvaluation(spark, test_File, tag_level)
nerDlEvaluation.computeAccuracyModel(ner_dl)
{% endhighlight %}

{% highlight scala %}
nerDl = NerDLModel.pretrained()

nerDlEvaluation = NerDLEvaluation(spark, testFile, tagLevel)
nerDlEvaluation.computeAccuracyModel(nerDl)
{% endhighlight %}

### Evaluating NER CRF

You can evaluate NER CRF when training an annotator.

- trainFile: Files with labeled NER entities for training. 
- modelPath: Path to save the model. When the path exists it loads the model instead of training it.
- testFile: Files with labeled NER entities. These files are used to evaluate the model. So, it's used for prediction and the labels as ground truth.
- format: The granularity of tagging when measuring accuracy on entities. Set "IOB" to include inside and beginning, empty to ignore it. For example
to display accuracy for entity I-PER and B-PER set "IOB" whereas just for entity PER set it as an empty string.

**Example:**
{% highlight python %}
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
{% endhighlight %}

{% highlight scala %}
val glove = new WordEmbeddings()
      .setInputCols("sentence", "token")
      .setOutputCol("glove")
      .setEmbeddingsSource("./glove.6B.100d.txt ", 100, WordEmbeddingsFormat.TEXT)
      .setCaseSensitive(true)

val nerTagger = new NerCrfApproach()
  .setInputCols(Array("sentence", "token","pos", "embeddings"))
  .setLabelColumn("label")
  .setOutputCol("ner")
  .setMaxEpochs(10)

val nerCrfEvaluation = new NerCrfEvaluation(testFile, format)
nerCrfEvaluation.computeAccuracyAnnotator(modelPath, trainFile, nerTagger, glove)
{% endhighlight %}

**Example for pretrained model:**
{% highlight python %}
ner_crf = NerCrfModel.pretrained()

nerCrfEvaluation = NerCrfEvaluation(spark, test_File, tag_level)
nerCrfEvaluation.computeAccuracyModel(ner_crf)
{% endhighlight %}

{% highlight scala %}
nerCrf = NerCrfModel.pretrained()

nerCrfEvaluation = NerCrfEvaluation(spark, testFile, tagLevel)
nerCrfEvaluation.computeAccuracyModel(nerCrf)
{% endhighlight %}

### Evaluating POS Tagger

You can evaluate POS either training an annotator or using a pretrained model.

- trainFile: A labeled POS file see and example [here](https://nlp.johnsnowlabs.com/docs/en/annotators#pos-dataset).
- testFile: A CoNLL-U format file.

**Example for annotator:**
{% highlight python %}
pos_tagger = PerceptronApproach() \
             .setInputCols(["document", "token"]) \
             .setOutputCol("pos") \
             .setNIterations(2)

posEvaluation = POSEvaluation(spark, test_file)
posEvaluation.computeAccuracyAnnotator(train_file, pos_tagger)
{% endhighlight %}

{% highlight scala %}
val posTagger = new PerceptronApproach()
      .setInputCols(Array("document", "token"))
      .setOutputCol("pos")
      .setNIterations(2)

val posEvaluation = new POSEvaluation(spark, testFile)
posEvaluation.computeAccuracyAnnotator(trainFile, posTagger)
{% endhighlight %}