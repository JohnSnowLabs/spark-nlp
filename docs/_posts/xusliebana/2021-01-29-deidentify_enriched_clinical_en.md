---
layout: model
title: Deidentify  (Enriched)
author: John Snow Labs
name: deidentify_enriched_clinical
date: 2021-01-29
task: De-identification
language: en
edition: Spark NLP for Healthcare 2.7.2
spark_version: 2.4
tags: [deidentify, en, obfuscation, licensed]
supported: true
annotator: DeIdentificationModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Deidentify (Large) is a deidentification model. It identifies instances of protected health information in text documents, and it can either obfuscate them (e.g., replacing names with different, fake names) or mask them (e.g., replacing “2020-06-04” with Some faker data). This model is useful for maintaining HIPAA compliance when dealing with text documents that contain protected health information.

## Predicted Entities

- PHONE
- PATIENT
- COUNTRY
- USERNAME
- LOCATION-OTHER
- DATE
- ID
- DOCTOR
- HOSPITAL
- IDNUM
- AGE
- MEDICALRECORD
- CITY
- FAX
- ZIP
- HEALTHPLAN
- PROFESSION
- BIOID
- URL
- EMAIL
- STATE
- ORGANIZATION
- STREET
- DEVICE

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/deidentify_enriched_clinical_en_2.7.2_2.4_1611917177874.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/deidentify_enriched_clinical_en_2.7.2_2.4_1611917177874.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol('text')\
    .setOutputCol('document')

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  	.setInputCols(["sentence", "token"])\
  	.setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
	.setInputCols(["sentence", "token", "embeddings"]) \
	.setOutputCol("ner")

ner_converter = NerConverter() \
  	.setInputCols(["sentence", "token", "ner"]) \
  	.setOutputCol("ner_chunk")

obfuscation = DeIdentificationModel.pretrained("deidentify_enriched_clinical", "en", "clinical/models") \
	.setInputCols(["sentence", "token", "ner_chunk"]) \
	.setOutputCol("obfuscated") \
	.setMode("obfuscate")

nlp_pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, obfuscation]) 

data = spark.createDataFrame([["""A . Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street"""]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)

```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  	.setInputCols(Array("sentence", "token"))
  	.setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
	.setInputCols(Array("sentence", "token", "embeddings"))
	.setOutputCol("ner")

val ner_converter = new NerConverter()
  	.setInputCols(Array("sentence", "token", "ner"))
  	.setOutputCol("ner_chunk")

val obfuscation = DeIdentificationModel.pretrained("deidentify_enriched_clinical", "en", "clinical/models")
	.setInputCols(Array("sentence", "token", "ner_chunk"))
	.setOutputCol("obfuscated")
	.setMode("obfuscate")

val nlpPipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, obfuscation))

val data = Seq("""A . Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street""").toDS.toDF("text")

val result = nlpPipeline.fit(data).transform(data)

```


{:.nlu-block}
```python
import nlu
nlu.load("en.de_identify.clinical").predict("""A . Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street""")
```

</div>

## Results

```bash
	sentence	deidentified
0	A .	A .
1	Record date : 2093-01-13 , David Hale , M.D .	Record date : 2093-01-18 , DR. Gregory Kaiser , M.D .
2	, Name : Hendrickson , Ora MR .	, Name : Joel Vasquez MR .
3	# 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09 .	# 67696 Date : 01/18/93 PCP : DR. Jennifer Eaton , 25 years-old , Record date : 2079-11-14 .
4	Cocke County Baptist Hospital .	San Leandro Hospital – San Leandro .
5	0295 Keats Street	3744 Retreat Avenue
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deidentify_enriched_clinical|
|Compatibility:|Spark NLP 2.7.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, ner_chunk]|
|Output Labels:|[deidentified]|
|Language:|en|
