---
layout: model
title: Typo Detector for Icelandic
author: John Snow Labs
name: distilbert_token_classifier_typo_detector
date: 2022-01-19
tags: [typo, distilbert, icelandic, token_classification, is, open_source]
task: Named Entity Recognition
language: is
edition: Spark NLP 3.3.4
spark_version: 3.0
supported: true
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` ([link](https://huggingface.co/m3hrdadfi/typo-detector-distilbert-is)) and it's been trained on a Icelandic synthetic data to detect typos, leveraging `DistilBERT` embeddings and `DistilBertForTokenClassification` for NER purposes. It classifies typo tokens as `PO`.

## Predicted Entities

`PO`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_typo_detector_is_3.3.4_3.0_1642599810600.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_typo_detector_is_3.3.4_3.0_1642599810600.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentenceDetector = SentenceDetector()\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")

tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_token_classifier_typo_detector", "is")\
  .setInputCols(["sentence",'token'])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")
      
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])
text = """Það er miög auðvelt að draga marktækar álykanir af texta með Spark NLP."""
data = spark.createDataFrame([[text]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```
```scala
val documentAssembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val sentenceDetector = SentenceDetector()
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_token_classifier_typo_detector", "is")
  .setInputCols(Array("sentence","token"))
  .setOutputCol("ner")

val ner_converter = NerConverter()
      .setInputCols(Array("sentence", "token", "ner"))
      .setOutputCol("ner_chunk")
      
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["Það er miög auðvelt að draga marktækar álykanir af texta með Spark NLP."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
+--------+---------+
|chunk   |ner_label|
+--------+---------+
|miög    |PO       |
|álykanir|PO       |
+--------+---------+
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
|Language:|is|
|Size:|505.7 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## Benchmarking

```bash
label         precision recall    f1-score  support
micro avg     0.98954   0.967603  0.978448  43800.0
macro-avg     0.98954   0.967603  0.978448  43800.0
weighted-avg  0.98954   0.967603  0.978448  43800.0
```