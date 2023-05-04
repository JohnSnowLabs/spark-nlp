---
layout: model
title: RoBertaZero-Shot Classification Base roberta_base_zero_shot_classifier_nli
author: John Snow Labs
name: roberta_base_zero_shot_classifier_nli
date: 2023-05-04
tags: [en, open_source, tensorflow]
task: Zero-Shot Classification
language: en
edition: Spark NLP 4.4.2
spark_version: [3.2, 3.0]
supported: true
engine: tensorflow
annotator: RoBertaForZeroShotClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is intended to be used for zero-shot text classification, especially in English. It is fine-tuned on NLI by using Roberta Base  model.

RoBertaForZeroShotClassificationusing a ModelForSequenceClassification trained on NLI (natural language inference) tasks. Equivalent of RoBertaForZeroShotClassification models, but these models don’t require a hardcoded number of potential classes, they can be chosen at runtime. It usually means it’s slower but it is much more flexible.

We used TFRobertaForSequenceClassification to train this model and used RoBertaForZeroShotClassification annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_zero_shot_classifier_nli_en_4.4.2_3.2_1683228241365.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_zero_shot_classifier_nli_en_4.4.2_3.2_1683228241365.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

zeroShotClassifier = RobertaForSequenceClassification \
.pretrained('roberta_base_zero_shot_classifier_nli', 'en') \
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

val zeroShotClassifier = RobertaForSequenceClassification.pretrained("roberta_base_zero_shot_classifier_nli", "en")
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
|Model Name:|roberta_base_zero_shot_classifier_nli|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[multi_class]|
|Language:|en|
|Size:|466.4 MB|
|Case sensitive:|true|