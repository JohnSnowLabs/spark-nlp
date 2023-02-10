---
layout: model
title: Extract Negation and Uncertainty Entities from Spanish Medical Texts (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_negation_uncertainty
date: 2022-08-11
tags: [es, clinical, licensed, token_classification, bert, ner, negation, uncertainty, linguistics]
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

This Named Entity Recognition model is intended for detecting relevant entities from Spanish medical texts and trained using the BertForTokenClassification method from the transformers library and [BERT based](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) embeddings. 
The model detects Negation Trigger (NEG), Negation Scope (NSCO), Uncertainty Trigger (UNC) and Uncertainty Scope (USCO).

## Predicted Entities

`NEG`, `NSCO`, `UNC`, `USCO`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_negation_uncertainty_es_4.0.2_3.0_1660231547751.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_negation_uncertainty_es_4.0.2_3.0_1660231547751.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_negation_uncertainty", "es", "clinical/models")\
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

data = spark.createDataFrame(["Con diagnóstico probable de cirrosis hepática (no conocida previamente) y peritonitis espontanea primaria con tratamiento durante 8 dias con ceftriaxona en el primer ingreso (no se realizó paracentesis control por escasez de liquido). Lesión tumoral en hélix izquierdo de 0,5 cms. de diámetro susceptible de ca basocelular perlado."], StringType()).toDF("text")
                              
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

val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_negation_uncertainty", "es", "clinical/models")
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

val data = Seq(Array("""Con diagnóstico probable de cirrosis hepática (no conocida previamente) y peritonitis espontanea primaria con tratamiento durante 8 dias con ceftriaxona en el primer ingreso (no se realizó paracentesis control por escasez de liquido). Lesión tumoral en hélix izquierdo de 0,5 cms. de diámetro susceptible de ca basocelular perlado.""")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+------------------------------------------------------+---------+
|chunk                                                 |ner_label|
+------------------------------------------------------+---------+
|probable                                              |UNC      |
|de cirrosis hepática                                  |USCO     |
|no                                                    |NEG      |
|conocida previamente                                  |NSCO     |
|no                                                    |NEG      |
|se realizó paracentesis control por escasez de liquido|NSCO     |
|susceptible de                                        |UNC      |
|ca basocelular perlado                                |USCO     |
+------------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_negation_uncertainty|
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

The model is prepared using the reference paper: "NLP applied to occupational health: MEDDOPROF shared task at IberLEF 2021 on automatic recognition, classification and normalization of professions and occupations from medical texts", Salvador Lima-Lopez, Eulalia Farr ́e-Maduell, Antonio Miranda-Escalada,
Vicent Briva-Iglesias and Martin Krallinger. Procesamiento del Lenguaje Natural, Revista nº 67, septiembre de 2021, pp. 243-256.

## Benchmarking

```bash
       label  precision    recall  f1-score   support
       B-NEG     0.9599    0.9667    0.9633      1833
       I-NEG     0.9216    0.9276    0.9246       152
       B-UNC     0.9040    0.8898    0.8968       508
       I-UNC     0.8772    0.8242    0.8499       182
      B-USCO     0.9164    0.8983    0.9073       708
      I-USCO     0.8473    0.8596    0.8534      2350
      B-NSCO     0.9475    0.9560    0.9517      2022
      I-NSCO     0.9323    0.9345    0.9334      5774
   micro-avg     0.9207    0.9239    0.9223     13529
   macro-avg     0.9133    0.9071    0.9101     13529
weighted-avg     0.9208    0.9239    0.9223     13529

```
