---
layout: model
title: Extract Demographic Entities from Social Determinants of Health Texts
author: John Snow Labs
name: ner_sdoh_demographics_wip
date: 2023-02-10
tags: [licensed, clinical, social_determinants, en, ner, demographics, sdoh, public_health]
task: Named Entity Recognition
language: en
nav_key: models
edition: Healthcare NLP 4.2.8
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts demographic information related to Social Determinants of Health from various kinds of biomedical documents.

## Predicted Entities

`Family_Member`, `Age`, `Gender`, `Geographic_Entity`, `Race_Ethnicity`, `Language`, `Spiritual_Beliefs`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_demographics_wip_en_4.2.8_3.0_1675998706136.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_demographics_wip_en_4.2.8_3.0_1675998706136.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_sdoh_demographics_wip", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector,
    tokenizer,
    clinical_embeddings,
    ner_model,
    ner_converter   
    ])

sample_texts = ["SOCIAL HISTORY: He is a former tailor from Korea.",
             "He lives alone,single and no children.",
             "Pt is a 61 years old married, Caucasian, Catholic woman. Pt speaks English reasonably well."]


data = spark.createDataFrame(sample_texts, StringType()).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_sdoh_demographics_wip", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(
    document_assembler, 
    sentence_detector,
    tokenizer,
    clinical_embeddings,
    ner_model,
    ner_converter   
))

val data = Seq("Pt is a 61 years old married, Caucasian, Catholic woman. Pt speaks English reasonably well.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------+-----+---+------------+
|ner_label        |begin|end|chunk       |
+-----------------+-----+---+------------+
|Gender           |16   |17 |He          |
|Geographic_Entity|43   |47 |Korea       |
|Gender           |0    |1  |He          |
|Family_Member    |29   |36 |children    |
|Age              |8    |19 |61 years old|
|Race_Ethnicity   |30   |38 |Caucasian   |
|Spiritual_Beliefs|41   |48 |Catholic    |
|Gender           |50   |54 |woman       |
|Language         |67   |73 |English     |
+-----------------+-----+---+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_sdoh_demographics_wip|
|Compatibility:|Healthcare NLP 4.2.8+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|858.4 KB|

## Benchmarking

```bash
	    label	     tp	    fp	   fn	   total	precision	   recall	       f1
              Age	 1346.0	  73.0	 74.0	  1420.0	 0.948555	 0.947887	 0.948221
Spiritual_Beliefs	  100.0	  13.0	 16.0	   116.0	 0.884956	 0.862069	 0.873362
    Family_Member	 4468.0	 134.0	 43.0	  4511.0	 0.970882	 0.990468	 0.980577
   Race_Ethnicity	   56.0	   0.0	 13.0	    69.0	 1.000000	 0.811594	 0.896000
           Gender	 9825.0	  67.0	247.0	 10072.0	 0.993227	 0.975477	 0.984272
Geographic_Entity	  225.0	   9.0	 29.0	   254.0	 0.961538	 0.885827	 0.922131
         Language	   51.0	   9.0	  5.0	    56.0	 0.850000	 0.910714	 0.879310
```
