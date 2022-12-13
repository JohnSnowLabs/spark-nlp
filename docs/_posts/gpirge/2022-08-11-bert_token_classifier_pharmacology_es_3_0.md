---
layout: model
title: Extract Pharmacological Entities From Spanish Medical Texts (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_pharmacology
date: 2022-08-11
tags: [es, clinical, licensed, token_classification, bert, ner, pharmacology]
task: Named Entity Recognition
language: es
edition: Healthcare NLP 4.0.2
spark_version: 3.0
supported: true
annotator: MedicalBertForTokenClassifier
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Named Entity Recognition model is intended for detecting pharmacological entities from Spanish medical texts and trained using the BertForTokenClassification method from the transformers library and [BERT based](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) embeddings. 
The model detects PROTEINAS and NORMALIZABLES.

## Predicted Entities

`PROTEINAS`, `NORMALIZABLES`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_pharmacology_es_4.0.2_3.0_1660236427687.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_pharmacology_es_4.0.2_3.0_1660236427687.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

tokenizer = Tokenizer()\
  .setInputCols("sentence")\
  .setOutputCol("token")

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_pharmacology", "es", "clinical/models")\
  .setInputCols("token", "sentence")\
  .setOutputCol("label")\
  .setCaseSensitive(True)

ner_converter = NerConverter()\
  .setInputCols(["sentence","token","label"])\
  .setOutputCol("ner_chunk")


pipeline =  Pipeline(stages=[
                      documentAssembler,
                      sentenceDetector,
                      tokenizer,
                      tokenClassifier,
                      ner_converter])

                          
data = spark.createDataFrame([["""Se realiza analítica destacando creatinkinasa 736 UI, LDH 545 UI, urea 63 mg/dl, CA 19.9 64,1 U/ml. Inmunofenotípicamente el tumor expresó vimentina, S-100, HMB-45 y actina. Se instauró el tratamiento con quimioterapia (Cisplatino, Interleukina II, Dacarbacina e Interferon alfa)."""]]).toDF("text")

result = pipeline.fit(data).transform(data)

```
```scala
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_pharmacology", "es", "clinical/models")
  .setInputCols(Array("token", "sentence"))
  .setOutputCol("label")
  .setCaseSensitive(True)

val ner_converter = new NerConverter()
  .setInputCols(Array("sentence","token","label"))
  .setOutputCol("ner_chunk")


val pipeline =  new Pipeline().setStages(Array(
                      documentAssembler,
                      sentenceDetector,
                      tokenizer,
                      tokenClassifier,
                      ner_converter))

val data = Seq(Array("Se realiza analítica destacando creatinkinasa 736 UI, LDH 545 UI, urea 63 mg/dl, CA 19.9 64,1 U/ml. Inmunofenotípicamente el tumor expresó vimentina, S-100, HMB-45 y actina. Se instauró el tratamiento con quimioterapia (Cisplatino, Interleukina II, Dacarbacina e Interferon alfa).")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+---------------+-------------+
|chunk          |ner_label    |
+---------------+-------------+
|creatinkinasa  |PROTEINAS    |
|LDH            |PROTEINAS    |
|urea           |NORMALIZABLES|
|CA 19.9        |PROTEINAS    |
|vimentina      |PROTEINAS    |
|S-100          |PROTEINAS    |
|HMB-45         |PROTEINAS    |
|actina         |PROTEINAS    |
|Cisplatino     |NORMALIZABLES|
|Interleukina II|PROTEINAS    |
|Dacarbacina    |NORMALIZABLES|
|Interferon alfa|PROTEINAS    |
+---------------+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_pharmacology|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|410.0 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## Benchmarking

```bash
          label  precision    recall  f1-score   support
B-NORMALIZABLES     0.9458    0.9694    0.9575      3076
I-NORMALIZABLES     0.8788    0.8969    0.8878       291
    B-PROTEINAS     0.9164    0.9369    0.9265      2234
    I-PROTEINAS     0.8825    0.7634    0.8186       748
      micro-avg     0.9257    0.9304    0.9280      6349
      macro-avg     0.9059    0.8917    0.8976      6349
   weighted-avg     0.9249    0.9304    0.9270      6349
```
