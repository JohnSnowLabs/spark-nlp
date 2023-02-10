---
layout: model
title: Extract Entities in Spanish Clinical Trial Abstracts (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_clinical_trials_abstracts
date: 2022-08-11
tags: [es, clinical, licensed, token_classification, bert, ner]
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

This Named Entity Recognition model is intended for detecting relevant entities from Spanish clinical trial abstracts and trained using the BertForTokenClassification method from the transformers library and [BERT based](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) embeddings. 
The model detects Pharmacological and Chemical Substances (CHEM), pathologies (DISO), and lab tests, diagnostic or therapeutic procedures (PROC).

## Predicted Entities

`CHEM`, `DISO`, `PROC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_clinical_trials_abstracts_es_4.0.2_3.0_1660229117151.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_clinical_trials_abstracts_es_4.0.2_3.0_1660229117151.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_clinical_trials_abstracts", "es", "clinical/models")\
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

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame(["""Efecto de la suplementación con ácido fólico sobre los niveles de homocisteína total en pacientes en hemodiálisis. La hiperhomocisteinemia es un marcador de riesgo independiente de morbimortalidad cardiovascular. Hemos prospectivamente reducir los niveles de homocisteína total (tHcy) mediante suplemento con ácido fólico y vitamina B6 (pp), valorando su posible correlación con dosis de diálisis, función  residual y parámetros nutricionales."""], StringType()).toDF("text")
                              
result = model.transform(data)

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

val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_clinical_trials_abstracts", "es", "clinical/models")
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

val data = Seq(Array("""Efecto de la suplementación con ácido fólico sobre los niveles de homocisteína total en pacientes en hemodiálisis. La hiperhomocisteinemia es un marcador de riesgo independiente de morbimortalidad cardiovascular. Hemos prospectivamente reducir los niveles de homocisteína total (tHcy) mediante suplemento con ácido fólico y vitamina B6 (pp), valorando su posible correlación con dosis de diálisis, función  residual y parámetros nutricionales.""")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------+---------+
|chunk                  |ner_label|
+-----------------------+---------+
|suplementación         |PROC     |
|ácido fólico           |CHEM     |
|niveles de homocisteína|PROC     |
|hemodiálisis           |PROC     |
|hiperhomocisteinemia   |DISO     |
|niveles de homocisteína|PROC     |
|tHcy                   |PROC     |
|ácido fólico           |CHEM     |
|vitamina B6            |CHEM     |
|pp                     |CHEM     |
|diálisis               |PROC     |
|función  residual      |PROC     |
+-----------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_clinical_trials_abstracts|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|410.0 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

The model is prepared using the reference paper: "A clinical trials corpus annotated with UMLS entities to enhance the access to evidence-based medicine", Leonardo Campillos-Llanos, Ana Valverde-Mateos, Adrián Capllonch-Carrión and Antonio Moreno-Sandoval. BMC Medical Informatics and Decision Making volume 21, Article number: 69 (2021)

## Benchmarking

```bash
       label  precision    recall  f1-score   support
      B-CHEM     0.9335    0.9314    0.9325      4944
      I-CHEM     0.8210    0.8689    0.8443      1251
      B-DISO     0.9406    0.9429    0.9417      5538
      I-DISO     0.9071    0.9115    0.9093      5129
      B-PROC     0.8850    0.9113    0.8979      5893
      I-PROC     0.8711    0.8615    0.8663      7047
   micro-avg     0.9010    0.9070    0.9040     29802
   macro-avg     0.8930    0.9046    0.8987     29802
weighted-avg     0.9012    0.9070    0.9040     29802

```
