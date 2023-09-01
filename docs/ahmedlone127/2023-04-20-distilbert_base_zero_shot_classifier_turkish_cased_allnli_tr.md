---
layout: model
title: DistilBERTZero-Shot Classification Base - distilbert_base_zero_shot_classifier_turkish_cased_allnli
author: John Snow Labs
name: distilbert_base_zero_shot_classifier_turkish_cased_allnli
date: 2023-04-20
tags: [distilbert, zero_shot, turkish, tr, base, open_source, tensorflow]
task: Zero-Shot Classification
language: tr
edition: Spark NLP 4.4.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: DistilBertForZeroShotClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is intended to be used for zero-shot text classification, especially in Trukish. It is fine-tuned on MNLI by using DistilBERT Base Uncased model.

DistilBertForZeroShotClassification using a ModelForSequenceClassification trained on NLI (natural language inference) tasks. Equivalent of DistilBertForSequenceClassification models, but these models don’t require a hardcoded number of potential classes, they can be chosen at runtime. It usually means it’s slower but it is much more flexible.

We used TFDistilBertForSequenceClassification to train this model and used DistilBertForZeroShotClassification annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_zero_shot_classifier_turkish_cased_allnli_tr_4.4.1_3.0_1682016415236.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_base_zero_shot_classifier_turkish_cased_allnli_tr_4.4.1_3.0_1682016415236.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
.pretrained('distilbert_base_zero_shot_classifier_turkish_cased_allnli', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512) \
.setCandidateLabels(["olumsuz", "olumlu"])

pipeline = Pipeline(stages=[
document_assembler,
tokenizer,
zeroShotClassifier
])
example = spark.createDataFrame([['Senaryo çok saçmaydı, beğendim diyemem.']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = Tokenizer()
.setInputCols("document")
.setOutputCol("token")

val zeroShotClassifier = DistilBertForZeroShotClassification.pretrained("distilbert_base_zero_shot_classifier_turkish_cased_allnli", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setCaseSensitive(true)
.setMaxSentenceLength(512)
.setCandidateLabels(Array("olumsuz", "olumlu"))

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, zeroShotClassifier))
val example = Seq("Senaryo çok saçmaydı, beğendim diyemem.").toDS.toDF("text")
val result = pipeline.fit(example).transform(example)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_zero_shot_classifier_turkish_cased_allnli|
|Compatibility:|Spark NLP 4.4.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[multi_class]|
|Language:|tr|
|Size:|254.3 MB|
|Case sensitive:|true|
