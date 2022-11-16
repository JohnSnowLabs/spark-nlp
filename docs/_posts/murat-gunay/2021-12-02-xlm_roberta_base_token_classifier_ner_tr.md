---
layout: model
title: Detect Person, Organization and Location in Turkish text
author: John Snow Labs
name: xlm_roberta_base_token_classifier_ner
date: 2021-12-02
tags: [xlm, roberta, ner, turkish, tr, open_source]
task: Named Entity Recognition
language: tr
edition: Spark NLP 3.3.2
spark_version: 2.4
supported: true
annotator: XlmRoBertaForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is imported from `Hugging Face-models`. This model is the fine-tuned version of "xlm-roberta-base" (a multilingual version of RoBERTa) using a reviewed version of well known Turkish NER dataset (https://github.com/stefan-it/turkish-bert/files/4558187/nerdata.txt)

## Predicted Entities

`PER`, `LOC`, `ORG`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_token_classifier_ner_tr_3.3.2_2.4_1638447262808.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_base_token_classifier_ner", "tr"))\
.setInputCols(["sentence",'token'])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
text = """Benim adım Cesur Yurttaş ve İstanbul'da yaşıyorum."""
result = model.transform(spark.createDataFrame([[text]]).toDF("text"))
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_base_token_classifier_ner", "tr"))\
.setInputCols(Array("sentence","token"))\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(Array("sentence", "token", "ner"))\
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["Benim adım Cesur Yurttaş ve İstanbul'da yaşıyorum."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("tr.ner.xlm_roberta").predict("""Benim adım Cesur Yurttaş ve İstanbul'da yaşıyorum.""")
```

</div>

## Results

```bash
+-------------+---------+
|chunk        |ner_label|
+-------------+---------+
|Cesur Yurttaş|PER      |
|İstanbul'da  |LOC      |
+-------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_token_classifier_ner|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|tr|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

[https://huggingface.co/akdeniz27/xlm-roberta-base-turkish-ner](https://huggingface.co/akdeniz27/xlm-roberta-base-turkish-ner)

## Benchmarking

```bash
accuracy: 0.9919343118732742

f1: 0.9492100796448622

precision: 0.9407349896480332

recall: 0.9578392621870883
```