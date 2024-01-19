---
layout: model
title: DeBerta Zero-Shot Classification Base - MNLI ANLI  (deberta_base_zero_shot_classifier_mnli_anli_v3
author: John Snow Labs
name: deberta_base_zero_shot_classifier_mnli_anli_v3
date: 2024-01-19
tags: [zero_shot, deberta, en, open_source, tensorflow]
task: Zero-Shot Classification
language: en
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: tensorflow
annotator: DeBertaForZeroShotClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaForZeroShotClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.deberta_base_zero_shot_classifier_mnli_anli_v3 is a English model originally trained by MoritzLaurer.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_base_zero_shot_classifier_mnli_anli_v3_en_5.2.4_3.0_1705688303164.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_base_zero_shot_classifier_mnli_anli_v3_en_5.2.4_3.0_1705688303164.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

zeroShotClassifier = DeBertaForZeroShotClassification \
.pretrained('deberta_base_zero_shot_classifier_mnli_anli_v3', 'en') \
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

val zeroShotClassifier = DeBertaForZeroShotClassification.pretrained("deberta_base_zero_shot_classifier_mnli_anli_v3", "en")
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
|Model Name:|deberta_base_zero_shot_classifier_mnli_anli_v3|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[multi_class]|
|Language:|en|
|Size:|441.1 MB|
|Case sensitive:|true|

## References

https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli