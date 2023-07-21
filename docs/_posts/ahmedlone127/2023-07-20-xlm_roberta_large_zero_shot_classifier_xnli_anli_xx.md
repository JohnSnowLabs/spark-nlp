---
layout: model
title: XlmRoBertaZero-Shot Classification Large xlm_roberta_large_zero_shot_classifier_xnli_anli
author: John Snow Labs
name: xlm_roberta_large_zero_shot_classifier_xnli_anli
date: 2023-07-20
tags: [zero_shot, xx, open_source, tensorflow]
task: Zero-Shot Classification
language: xx
edition: Spark NLP 5.0.2
spark_version: 3.0
supported: true
engine: tensorflow
annotator: XlmRoBertaForZeroShotClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is intended to be used for zero-shot text classification, especially in English. It is fine-tuned on NLI by using XlmRoberta Large model.

XlmRoBertaForZeroShotClassificationusing a ModelForSequenceClassification trained on NLI (natural language inference) tasks. Equivalent of TFXLMRoBertaForZeroShotClassification models, but these models don’t require a hardcoded number of potential classes, they can be chosen at runtime. It usually means it’s slower but it is much more flexible.

We used TFXLMRobertaForSequenceClassification to train this model and used XlmRoBertaForZeroShotClassification annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_zero_shot_classifier_xnli_anli_xx_5.0.2_3.0_1689886974932.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_zero_shot_classifier_xnli_anli_xx_5.0.2_3.0_1689886974932.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

zeroShotClassifier = XlmRobertaForSequenceClassification \
.pretrained('xlm_roberta_large_zero_shot_classifier_xnli_anli', 'xx') \
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

val zeroShotClassifier = XlmRobertaForSequenceClassification.pretrained("xlm_roberta_large_zero_shot_classifier_xnli_anli", "xx")
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
|Model Name:|xlm_roberta_large_zero_shot_classifier_xnli_anli|
|Compatibility:|Spark NLP 5.0.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[label]|
|Language:|xx|
|Size:|2.0 GB|
|Case sensitive:|true|