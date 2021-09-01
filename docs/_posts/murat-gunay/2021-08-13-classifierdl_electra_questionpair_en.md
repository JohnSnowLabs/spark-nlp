---
layout: model
title: Question Pair Classifier
author: John Snow Labs
name: classifierdl_electra_questionpair
date: 2021-08-13
tags: [en, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.1.3
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Identifies whether two question sentences are semantically repetitive or different.

## Predicted Entities

`almost_same`, `not_same`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_electra_questionpair_en_3.1.3_2.4_1628840750568.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

- The model is trained with `sent_electra_large_uncased` embeddings therefore the same embeddings should be used in the prediction pipeline.

- The question pairs should be identified with "q1" and "q2" in the text. The input text format should be as follows : `text = "q1: What is your name? q2: Who are you?"`

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

embeddings = BertSentenceEmbeddings.pretrained("sent_electra_large_uncased", "en") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")

document_classifier = ClassifierDLModel.load('outputs/_models/quora_question_pair_classifier_9') \
  .setInputCols(["document", "sentence_embeddings"]) \
  .setOutputCol("class")

nlpPipeline = Pipeline(stages=[document, embeddings, document_classifier])
light_pipeline = LightPipeline(nlpPipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result_1 = light_pipeline.annotate("q1: What is your favorite movie? q2: Which movie do you like most?")
print(result_1["class"])

result_2 = light_pipeline.annotate("q1: What is your favorite movie? q2: Which movie genre would you like to watch?")
print(result_2["class"])
```
```scala
val document = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val embeddings = BertSentenceEmbeddings.pretrained("sent_electra_large_uncased", "en")
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

val document_classifier = ClassifierDLModel.load("outputs/_models/quora_question_pair_classifier_9")
  .setInputCols(Array("document", "sentence_embeddings"))
  .setOutputCol("class")

val nlpPipeline = new Pipeline().setStages(Array(document, embeddings, document_classifier))
val light_pipeline = LightPipeline(nlpPipeline.fit(spark.createDataFrame([['']]).toDF("text")))

val result_1 = light_pipeline.annotate("q1: What is your favorite movie? q2: Which movie do you like most?")

val result_2 = light_pipeline.annotate("q1: What is your favorite movie? q2: Which movie genre would you like to watch?")
```
</div>

## Results

```bash
['almost_same']
['not_same']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_electra_questionpair|
|Compatibility:|Spark NLP 3.1.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|

## Data Source

A custom dataset is used based on this source : "https://www.kaggle.com/c/quora-question-pairs/data".

## Benchmarking

```bash
              precision    recall  f1-score   support

 almost_same       0.85      0.91      0.88     29652
    not_same       0.90      0.84      0.87     29634

    accuracy                           0.88     59286
   macro avg       0.88      0.88      0.88     59286
weighted avg       0.88      0.88      0.88     59286
```