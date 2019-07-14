---
layout: article
title: Evaluation
permalink: /docs/en/evaluation
key: docs-evaluation
modify_date: "2019-07-14"
---

## Spark NLP Evaluation
This module includes tools to evaluate the accuracy of annotators. It includes specific metrics for each **annotator** and its training time.
The results will display on the console or to an MLflow run.

Just whit a simple import you can start using it.

**Example:**
{% highlight scala %}
import com.johnsnowlabs.nlp.eval._
{% endhighlight %}

**Note:** Currently working just for scala.

### Evaluating Norvig Spell Checker
You can evaluate this spell checker either training an annotator or using a pretrained model.

**Example for annotator:**
 
{% highlight scala %}
val spell = new NorvigSweetingApproach()
   .setInputCols(Array("token"))
   .setOutputCol("checked")
   .setDictionary(dictionaryFile)

val norvigSpellEvaluation = new NorvigSpellEvaluation(testFile, groundTruthFile)
norvigSpellEvaluation.computeAccuracyAnnotator(trainFile, spell)
{% endhighlight %}

**Example for pretrained model:**
{% highlight scala %}
val spell = NorvigSweetingModel.pretrained()
val norvigSpellEvaluation = new NorvigSpellEvaluation(testFile, groundTruthFile)
norvigSpellEvaluation.computeAccuracyModel(spell)
{% endhighlight %}

### Evaluating Symmetric Spell Checker
You can evaluate this spell checker either training an annotator or using a pretrained model.

**Example for annotator:**
 
{% highlight scala %}
val spell = new SymmetricDeleteApproach()
      .setInputCols(Array("token"))
      .setOutputCol("checked")
      .setDictionary(dictionaryFile)

val symSpellEvaluation = new SymSpellEvaluation(testFile, groundTruthFile)
    symSpellEvaluation.computeAccuracyAnnotator(trainFile, spell)
{% endhighlight %}

**Example for pretrained model:**
{% highlight scala %}
val spell = SymmetricDeleteModel.pretrained()
val symSpellEvaluation = new SymSpellEvaluation(testFile, groundTruthFile)
symSpellEvaluation.computeAccuracyModel(spell)
{% endhighlight %}

### Evaluating NER DL
You can evaluate NER DL when training an annotator.

**Example:**
{% highlight scala %}
val glove = new WordEmbeddings()
      .setInputCols("sentence", "token")
      .setOutputCol("glove")
      .setEmbeddingsSource("glove.6B.100d.txt ",
        100, WordEmbeddingsFormat.TEXT)
      .setCaseSensitive(true)

val nerTagger = new NerDLApproach()
  .setInputCols(Array("sentence", "token", "glove"))
  .setLabelColumn("label")
  .setOutputCol("ner")
  .setMaxEpochs(10)

val nerDLEvaluation = new NerDLEvaluation(testFiles, format)
    nerDLEvaluation.computeAccuracyAnnotator(modelPath, trainFile, nerTagger, glove)
{% endhighlight %}

### Evaluating NER CRF
You can evaluate NER CRF when training an annotator.
**Example:**
{% highlight scala %}
val glove = new WordEmbeddings()
      .setInputCols("sentence", "token")
      .setOutputCol("glove")
      .setEmbeddingsSource("./glove.6B.100d.txt ",
        100, WordEmbeddingsFormat.TEXT)
      .setCaseSensitive(true)

val nerTagger = new NerCrfApproach()
  .setInputCols(Array("sentence", "token","pos", "glove"))
  .setLabelColumn("label")
  .setOutputCol("ner")
  .setMaxEpochs(10)

val nerCrfEvaluation = new NerCrfEvaluation(testFiles, format)
    nerCrfEvaluation.computeAccuracyAnnotator(modelPath, trainFile, nerTagger, glove)
{% endhighlight %}



