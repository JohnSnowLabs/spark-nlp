---
layout: model
title: Extract Entities in Spanish Clinical Trial Abstracts
author: John Snow Labs
name: ner_clinical_trials_abstracts
date: 2022-08-12
tags: [es, clinical, licensed, ner, clinical_abstracts, chem, diso, proc]
task: Named Entity Recognition
language: es
edition: Healthcare NLP 4.0.2
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Named Entity Recognition model detects relevant entities from Spanish clinical trial abstracts and trained by using MedicalNerApproach annotator that allows to train generic NER models based on Neural Networks. 
The model detects CHEM (Pharmacological and Chemical Substances), pathologies (DISO), and lab tests, diagnostic or therapeutic procedures (PROC).

## Predicted Entities

`CHEM`, `DISO`, `PROC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_trials_abstracts_es_4.0.2_3.0_1660339167613.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_clinical_trials_abstracts_es_4.0.2_3.0_1660339167613.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner = MedicalNerModel.pretrained('ner_clinical_trials_abstracts', "es", "clinical/models") \
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

data = spark.createDataFrame([["""Efecto de la suplementación con ácido fólico sobre los niveles de homocisteína total en pacientes en hemodiálisis. La hiperhomocisteinemia es un marcador de riesgo independiente de morbimortalidad cardiovascular. Hemos prospectivamente reducir los niveles de homocisteína total (tHcy) mediante suplemento con ácido fólico y vitamina B6 (pp), valorando su posible correlación con dosis de diálisis, función  residual y parámetros nutricionales."""]]).toDF("text")

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
	.setInputCols(["sentence","token"])
	.setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_clinical_trials_abstracts", "es", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documenter, sentenceDetector, tokenizer, word_embeddings, ner_model, ner_converter))

val data = Seq(Array("""Efecto de la suplementación con ácido fólico sobre los niveles de homocisteína total en pacientes en hemodiálisis. La hiperhomocisteinemia es un marcador de riesgo independiente de morbimortalidad cardiovascular. Hemos prospectivamente reducir los niveles de homocisteína total (tHcy) mediante suplemento con ácido fólico y vitamina B6 (pp), valorando su posible correlación con dosis de diálisis, función  residual y parámetros nutricionales.""")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------------+---------+
|chunk                        |ner_label|
+-----------------------------+---------+
|suplementación               |PROC     |
|ácido fólico                 |CHEM     |
|niveles de homocisteína      |PROC     |
|hemodiálisis                 |PROC     |
|hiperhomocisteinemia         |DISO     |
|niveles de homocisteína total|PROC     |
|tHcy                         |PROC     |
|ácido fólico                 |CHEM     |
|vitamina B6                  |CHEM     |
|pp                           |CHEM     |
|diálisis                     |PROC     |
|función  residual            |PROC     |
+-----------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_trials_abstracts|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|16.4 MB|

## References

The model is prepared using the reference paper: "A clinical trials corpus annotated with UMLS entities to enhance the access to evidence-based medicine", Leonardo Campillos-Llanos, Ana Valverde-Mateos, Adrián Capllonch-Carrión and Antonio Moreno-Sandoval. BMC Medical Informatics and Decision Making volume 21, Article number: 69 (2021)

## Benchmarking

```bash
       label  precision    recall  f1-score   support
      B-DISO       0.91      0.93      0.92      2465
      I-DISO       0.85      0.88      0.86      2788
      B-CHEM       0.91      0.92      0.91      1558
      I-CHEM       0.82      0.91      0.86       645
      B-PROC       0.89      0.91      0.90      3348
      I-PROC       0.80      0.87      0.83      4232
   micro-avg       0.86      0.90      0.88     15036
   macro-avg       0.86      0.90      0.88     15036
weighted-avg       0.86      0.90      0.88     15036
```