---
layout: model
title: DistilBERTZero-Shot Classification Base - MNLI(distilbert_base_zero_shot_classifier_uncased_mnli
author: John Snow Labs
name: distilbert_base_zero_shot_classifier_uncased_mnli
date: 2024-08-03
tags: [en, zero_shot, distilbert, mnli, open_source, onnx]
task: Zero-Shot Classification
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: InstructorEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is intended to be used for zero-shot text classification, especially in English. It is fine-tuned on MNLI by using DistilBERT Base Uncased model.

DistilBertForZeroShotClassification using a ModelForSequenceClassification trained on NLI (natural language inference) tasks. Equivalent of DistilBertForSequenceClassification models, but these models don’t require a hardcoded number of potential classes, they can be chosen at runtime. It usually means it’s slower but it is much more flexible.

We used TFDistilBertForSequenceClassification to train this model and used DistilBertForZeroShotClassification annotator in Spark NLP 🚀 for prediction at scale!

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_zero_shot_classifier_uncased_mnli_en_5.4.2_3.0_1722682614871.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_base_zero_shot_classifier_uncased_mnli_en_5.4.2_3.0_1722682614871.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler() \
.setInputCol('text') \
.setOutputCol('document')

tokenizer = Tokenizer() \
.setInputCols(['document']) \
.setOutputCol('token')

zeroShotClassifier = DistilBertForZeroShotClassification \
.pretrained('distilbert_base_zero_shot_classifier_uncased_mnli', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512) \
.setCandidateLabels(["urgent", "mobile", "travel", "movie", "music", "sport", "weather", "technology"])

pipeline = Pipeline(stages=[
document_assembler,
tokenizer,
zeroShotClassifier
])

example = spark.createDataFrame([['I have a problem with my iphone that needs to be resolved asap!!']]).toDF("text")
result = pipeline.fit(example).transform(example)

```
```scala

val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = Tokenizer()
.setInputCols("document")
.setOutputCol("token")

val zeroShotClassifier = DistilBertForZeroShotClassification.pretrained("distilbert_base_zero_shot_classifier_uncased_mnli", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setCaseSensitive(true)
.setMaxSentenceLength(512)
.setCandidateLabels(Array("urgent", "mobile", "travel", "movie", "music", "sport", "weather", "technology"))

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, zeroShotClassifier))
val example = Seq("I have a problem with my iphone that needs to be resolved asap!!").toDS.toDF("text")
val result = pipeline.fit(example).transform(example)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_zero_shot_classifier_uncased_mnli|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[instructor]|
|Language:|en|
|Size:|406.0 MB|