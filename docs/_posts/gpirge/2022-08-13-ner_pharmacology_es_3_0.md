---
layout: model
title: Extract Pharmacological Entities from Spanish Medical Texts
author: John Snow Labs
name: ner_pharmacology
date: 2022-08-13
tags: [es, clinical, licensed, ner, pharmacology, proteinas, normalizables]
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

This Named Entity Recognition model is intended for detecting pharmacological entities from Spanish medical texts and trained by using MedicalNerApproach annotator that allows to train generic NER models based on Neural Networks.. 
The model detects PROTEINAS and NORMALIZABLES.

## Predicted Entities

`PROTEINAS`, `NORMALIZABLES`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_pharmacology_es_4.0.2_3.0_1660355686728.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_pharmacology_es_4.0.2_3.0_1660355686728.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner = MedicalNerModel.pretrained('ner_pharmacology', "es", "clinical/models") \
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

data = spark.createDataFrame([["""Se realiza analítica destacando creatinkinasa 736 UI, LDH 545 UI, urea 63 mg/dl, CA 19.9 64,1 U/ml. Inmunofenotípicamente el tumor expresó vimentina, S-100, HMB-45 y actina. Se instauró el tratamiento con quimioterapia (Cisplatino, Interleukina II, Dacarbacina e Interferon alfa)."""]]).toDF("text")

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

val ner_model = MedicalNerModel.pretrained("ner_pharmacology", "es", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")

val ner_converter = new NerConverter()
  .setInputCols(Array("sentence", "token", "ner"))
  .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documenter, sentenceDetector, tokenizer, word_embeddings, ner_model, ner_converter))

val data = Seq(Array("""Se realiza analítica destacando creatinkinasa 736 UI, LDH 545 UI, urea 63 mg/dl, CA 19.9 64,1 U/ml. Inmunofenotípicamente el tumor expresó vimentina, S-100, HMB-45 y actina. Se instauró el tratamiento con quimioterapia (Cisplatino, Interleukina II, Dacarbacina e Interferon alfa).""")).toDS().toDF("text")

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
|Model Name:|ner_pharmacology|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|16.3 MB|

## References

The model is prepared using the reference paper: "NLP applied to occupational health: MEDDOPROF shared task at IberLEF 2021 on automatic recognition, classification and normalization of professions and occupations from medical texts", Salvador Lima-Lopez, Eulalia Farr ́e-Maduell, Antonio Miranda-Escalada,
Vicent Briva-Iglesias and Martin Krallinger. Procesamiento del Lenguaje Natural, Revista nº 67, septiembre de 2021, pp. 243-256.

## Benchmarking

```bash
          label  precision    recall  f1-score   support
    B-PROTEINAS       0.88      0.93      0.90       813
    I-PROTEINAS       0.83      0.71      0.77       321
B-NORMALIZABLES       0.94      0.93      0.93       954
I-NORMALIZABLES       0.87      0.84      0.86       134
      micro-avg       0.90      0.89      0.90      2222
      macro-avg       0.88      0.85      0.86      2222
   weighted-avg       0.90      0.89      0.89      2222
```
