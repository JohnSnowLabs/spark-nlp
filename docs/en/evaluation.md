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