---
layout: model
title: TREC(6) Question Classifier
author: John Snow Labs
name: classifierdl_use_trec6
date: 2021-01-08
tags: [classifier, open_source, en, text_classification]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify open-domain, fact-based questions into one of the following broad semantic categories: Abbreviation, Description, Entities, Human Beings, Locations, or Numeric Values.

## Predicted Entities

``ABBR``,  ``DESC``,  ``NUM``,  ``ENTY``,  ``LOC``,  ``HUM``.

{:.btn-box}
[Live Demo](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_TREC.ipynb){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_trec6_en_2.7.1_2.4_1610118062425.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

use = UniversalSentenceEncoder.pretrained(lang="en") \
  .setInputCols(["document"])\
  .setOutputCol("sentence_embeddings")

document_classifier = ClassifierDLModel.pretrained('classifierdl_use_trec6', 'en') \
  .setInputCols(["document", "sentence_embeddings"]) \
  .setOutputCol("class")

nlpPipeline = Pipeline(stages=[documentAssembler, use, document_classifier])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate('When did the construction of stone circles begin in the UK?')
```
```scala
val documentAssembler = DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")
val use = UniversalSentenceEncoder.pretrained(lang="en")
  .setInputCols(Array("document"))
  .setOutputCol("sentence_embeddings")

val document_classifier = ClassifierDLModel.pretrained("classifierdl_use_trec6", "en")
  .setInputCols(Array("document", "sentence_embeddings"))
  .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, use, document_classifier))

val result = pipeline.fit(Seq.empty["When did the construction of stone circles begin in the UK?"].toDS.toDF("text")).transform(data)
```
</div>

## Results

```bash
+------------------------------------------------------------------------------------------------+------------+
|document                                                                                        |class       |
+------------------------------------------------------------------------------------------------+------------+
|When did the construction of stone circles begin in the UK?                                     | NUM        |
+------------------------------------------------------------------------------------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_use_trec6|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|

## Benchmarking

```bash
              precision    recall  f1-score   support

        ABBR       0.00      0.00      0.00        26
        DESC       0.89      0.96      0.92       343
        ENTY       0.86      0.86      0.86       391
         HUM       0.91      0.90      0.91       366
         LOC       0.88      0.91      0.89       233
         NUM       0.94      0.94      0.94       274

    accuracy                           0.89      1633
   macro avg       0.75      0.76      0.75      1633
weighted avg       0.88      0.89      0.89      1633
```