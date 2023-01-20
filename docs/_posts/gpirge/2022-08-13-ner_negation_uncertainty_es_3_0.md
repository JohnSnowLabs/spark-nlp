---
layout: model
title: Extract Negation and Uncertainty Entities from Spanish Medical Texts
author: John Snow Labs
name: ner_negation_uncertainty
date: 2022-08-13
tags: [es, clinical, licensed, ner, unc, usco, neg, nsco, negation, uncertainty]
task: Named Entity Recognition
language: es
edition: Healthcare NLP 4.0.2
spark_version: 3.0
supported: true
recommended: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Named Entity Recognition model is intended for detecting relevant entities from Spanish medical texts and trained by using MedicalNerApproach annotator that allows to train generic NER models based on Neural Networks. 
The model detects Negation Trigger (NEG), Negation Scope (NSCO), Uncertainty Trigger (UNC) and Uncertainty Scope (USCO).

## Predicted Entities

`NEG`, `UNC`, `USCO`, `NSCO`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_negation_uncertainty_es_4.0.2_3.0_1660357762363.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_negation_uncertainty_es_4.0.2_3.0_1660357762363.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")
 
sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
	.setInputCols(["document"])\
	.setOutputCol("sentence")

tokenizer = Tokenizer()\
	.setInputCols(["sentence"])\
	.setOutputCol("token")

word_embeddings = RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("embeddings")

ner = MedicalNerModel.pretrained('ner_negation_uncertainty', "es", "clinical/models") \
	.setInputCols(["sentence", "token", "embeddings"]) \
	.setOutputCol("ner")
 
ner_converter = NerConverter()\
	.setInputCols(["sentence", "token", "ner"])\
	.setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
	document_assembler,
	sentenceDetectorDL,
	tokenizer,
	word_embeddings,
	ner,
	ner_converter])

data = spark.createDataFrame([["""Con diagnóstico probable de cirrosis hepática (no conocida previamente) y peritonitis espontanea primaria con tratamiento durante 8 dias con ceftriaxona en el primer ingreso (no se realizó paracentesis control por escasez de liquido). Lesión tumoral en hélix izquierdo de 0,5 cms. de diámetro susceptible de ca basocelular perlado."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documenter = new DocumentAssembler() 
  .setInputCol("text") 
  .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val word_embeddings = RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")
  .setInputCols(Array("sentence","token"))
  .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_negation_uncertainty", "es", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")

val ner_converter = new NerConverter()
  .setInputCols(Array("sentence", "token", "ner"))
  .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documenter, sentenceDetector, tokenizer, word_embeddings, ner_model, ner_converter))

val data = Seq(Array("""Con diagnóstico probable de cirrosis hepática (no conocida previamente) y peritonitis espontanea primaria con tratamiento durante 8 dias con ceftriaxona en el primer ingreso (no se realizó paracentesis control por escasez de liquido). Lesión tumoral en hélix izquierdo de 0,5 cms. de diámetro susceptible de ca basocelular perlado.""")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+------------------------------------------------------+---------+
|chunk                                                 |ner_label|
+------------------------------------------------------+---------+
|probable de                                           |UNC      |
|cirrosis hepática                                     |USCO     |
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
|Model Name:|ner_negation_uncertainty|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|16.2 MB|

## References

The model is prepared using the reference paper: "NLP applied to occupational health: MEDDOPROF shared task at IberLEF 2021 on automatic recognition, classification and normalization of professions and occupations from medical texts", Salvador Lima-Lopez, Eulalia Farr ́e-Maduell, Antonio Miranda-Escalada,
Vicent Briva-Iglesias and Martin Krallinger. Procesamiento del Lenguaje Natural, Revista nº 67, septiembre de 2021, pp. 243-256.

## Benchmarking

```bash
       label  precision    recall  f1-score   support
       B-NEG       0.93      0.97      0.95      1409
       I-NEG       0.80      0.90      0.85       119
       B-UNC       0.82      0.85      0.83       395
       I-UNC       0.77      0.78      0.77       166
      B-USCO       0.76      0.79      0.77       394
      I-USCO       0.61      0.81      0.69      1468
      B-NSCO       0.92      0.92      0.92      1308
      I-NSCO       0.87      0.89      0.88      3806
   micro-avg       0.82      0.89      0.85      9065
   macro-avg       0.81      0.86      0.83      9065
weighted-avg       0.83      0.89      0.86      9065
```
