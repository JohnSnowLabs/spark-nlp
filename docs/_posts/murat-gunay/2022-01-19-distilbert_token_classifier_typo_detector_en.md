---
layout: model
title: Typo Detector
author: John Snow Labs
name: distilbert_token_classifier_typo_detector
date: 2022-01-19
tags: [typo, distilbert, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.3.4
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` ([link](https://huggingface.co/m3hrdadfi/typo-detector-distilbert-en)) and it's been trained on NeuSpell corpus to detect typos, leveraging `DistilBERT` embeddings and `DistilBertForTokenClassification` for NER purposes. It classifies typo tokens as `PO`.

## Predicted Entities

`PO`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_typo_detector_en_3.3.4_3.0_1642581005021.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_typo_detector_en_3.3.4_3.0_1642581005021.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_token_classifier_typo_detector", "en")\
.setInputCols(["sentence",'token'])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])
text = """He had also stgruggled with addiction during his tine in Congress."""
data = spark.createDataFrame([[text]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_token_classifier_typo_detector", "en")
.setInputCols(Array("sentence","token"))
.setOutputCol("ner")

val ner_converter = NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["He had also stgruggled with addiction during his tine in Congress."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.typos.distilbert").predict("""He had also stgruggled with addiction during his tine in Congress.""")
```

</div>

## Results

```bash
+------------+---------+
|chunk       |ner_label|
+------------+---------+
|stgruggled  |PO       |
|tine        |PO       |
+------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_token_classifier_typo_detector|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|244.1 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## Data Source

[https://github.com/neuspell/neuspell](https://github.com/neuspell/neuspell)

## Benchmarking

```bash
label        precision  recall    f1-score  support
micro-avg    0.992332   0.985997  0.989154  416054.0
macro-avg    0.992332   0.985997  0.989154  416054.0
weighted-avg 0.992332   0.985997  0.989154  416054.0
```
