---
layout: model
title: Detect Posology concepts (ner_posology_healthcare)
author: John Snow Labs
name: ner_posology_healthcare
date: 2021-04-01
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Detect Drug, Dosage and administration instructions in text using pretraiend NER model.


## Predicted Entities


`Drug`, `Duration`, `Strength`, `Form`, `Frequency`, `Dosage`, `Route`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_POSOLOGY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_healthcare_en_3.0.0_3.0_1617260847574.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models") \
	.setInputCols(["document"]) \
	.setOutputCol("sentence")

tokenizer = Tokenizer()\
	.setInputCols(["sentence"])\
	.setOutputCol("token")

embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_posology_healthcare", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("entities")

pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["The patient is a 40-year-old white male who presents with a chief complaint of 'chest pain'. The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that chest pain started yesterday evening.  He has been advised Aspirin 81 milligrams QDay. insulin 50 units in a.m. HCTZ 50 mg QDay. Nitroglycerin 1/150 sublingually."]]).toDF("text"))
```
```scala

val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
	.setInputCols("document")
	.setOutputCol("sentence")

val tokenizer = new Tokenizer()
	.setInputCols("sentence")
	.setOutputCol("token")

val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_posology_healthcare", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val text = """The patient is a 40-year-old white male who presents with a chief complaint of 'chest pain'. The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that chest pain started yesterday evening.  He has been advised Aspirin 81 milligrams QDay. insulin 50 units in a.m. HCTZ 50 mg QDay. Nitroglycerin 1/150 sublingually."""

val data = Seq(text).toDS.toDF("text")

val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}

```python
import nlu
nlu.load("en.med_ner.posology.healthcare").predict("""Put your text here.""")
```

</div>

## Results

```bash
+-------------+---------+
|chunk        |ner_label|
+-------------+---------+
|Aspirin      |Drug     |
|81 milligrams|Strength |
|QDay         |Frequency|
|insulin      |Drug     |
|50 units     |Dosage   |
|in a.m.      |Frequency|
|HCTZ         |Drug     |
|50 mg        |Strength |
|QDay         |Frequency|
|Nitroglycerin|Drug     |
|1/150        |Strength |
|sublingually.|Route    |
+-------------+---------+
```

{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_posology_healthcare|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|




## Benchmarking
```bash
label       tp      fp     fn     total      precision  recall  f1
DURATION    995.0   463.0  132.0  1127.0     0.6824     0.8829  0.7698
DRUG        4957.0  632.0  476.0  5433.0     0.8869     0.9124  0.8995
DOSAGE      539.0   183.0  380.0   919.0     0.7465     0.5865  0.6569
ROUTE       676.0   47.0   129.0   805.0      0.935     0.8398  0.8848
FREQUENCY   3688.0  675.0  313.0  4001.0     0.8453     0.9218  0.8819
FORM        1328.0  261.0  294.0  1622.0     0.8357     0.8187  0.8272
STRENGTH    5008.0  687.0  557.0  5565.0     0.8794     0.8999  0.8895
macro-avg     -       -      -       -         -          -     0.82994
micro-avg     -       -      -       -         -          -     0.86743
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA3NjIzOTY1XX0=
-->