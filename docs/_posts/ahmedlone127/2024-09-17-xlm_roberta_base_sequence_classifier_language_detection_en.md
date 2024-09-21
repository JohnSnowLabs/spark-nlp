---
layout: model
title: XLM-RoBERTa Sequence Classification Base - Language Detection (xlm_roberta_base_sequence_classifier_language_detection)
author: John Snow Labs
name: xlm_roberta_base_sequence_classifier_language_detection
date: 2024-09-17
tags: [sequence_classification, roberta, openvino, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: openvino
annotator: XlmRoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

“
XLM-RoBERTa Model with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

xlm_roberta_base_sequence_classifier_language_detection is a fine-tuned XLM-RoBERTa model that is ready to be used for Sequence Classification tasks such as sentiment analysis or multi-class text classification and it achieves state-of-the-art performance.

We used TFXLMRobertaForSequenceClassification to train this model and used XlmRoBertaForSequenceClassification annotator in Spark NLP 🚀 for prediction at scale!

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_sequence_classifier_language_detection_en_5.5.0_3.0_1726563005096.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_sequence_classifier_language_detection_en_5.5.0_3.0_1726563005096.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
.setInputCol('text') \
.setOutputCol('document')

tokenizer = Tokenizer() \
.setInputCols(['document']) \
.setOutputCol('token')

sequenceClassifier = XlmRoBertaForSequenceClassification \
.pretrained('xlm_roberta_base_sequence_classifier_language_detection', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler,
tokenizer,
sequenceClassifier
])

example = spark.createDataFrame([['I really liked that movie!']]).toDF("text")
result = pipeline.fit(example).transform(example)

```
```scala

		val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = Tokenizer()
.setInputCols("document")
.setOutputCol("token")

val tokenClassifier = XlmRoBertaForSequenceClassification.pretrained("xlm_roberta_base_sequence_classifier_language_detection", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("I really liked that movie!").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_sequence_classifier_language_detection|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[label]|
|Language:|en|
|Size:|870.5 MB|
|Case sensitive:|true|