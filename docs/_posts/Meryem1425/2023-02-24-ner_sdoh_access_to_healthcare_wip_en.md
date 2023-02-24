---
layout: model
title: Extract Access to Healthcare Entities from Social Determinants of Health Texts
author: John Snow Labs
name: ner_sdoh_access_to_healthcare_wip
date: 2023-02-24
tags: [licensed, clinical, en, social_determinants, ner, sdoh, public_health, access, healthcare, access_to_healthcare]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.1
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts access to healthcare information related to Social Determinants of Health from various kinds of biomedical documents.

## Predicted Entities

`Insurance_Status`, `Healthcare_Institution`, `Access_To_Care`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_access_to_healthcare_wip_en_4.3.1_3.0_1677202491556.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_access_to_healthcare_wip_en_4.3.1_3.0_1677202491556.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = MedicalNerModel.pretrained("ner_sdoh_access_to_healthcare_wip", "en", "clinical/models")\
    .setInputCols(["sentence", "token","embeddings"])\
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

sample_texts = ["She has a pension and private health insurance, she reports feeling lonely and isolated.",
             "He also reported food insecurityduring his childhood and lack of access to adequate healthcare."]


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

val ner_model = MedicalNerModel.pretrained("ner_sdoh_access_to_healthcare_wip", "en", "clinical/models")
    .setInputCols(Array("sentence", "token","embeddings"))
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

val data = Seq("She has a pension and private health insurance, she reports feeling lonely and isolated.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------------+-----+---+----------------+
|chunk                        |begin|end|ner_label       |
+-----------------------------+-----+---+----------------+
|private health insurance     |22   |45 |Insurance_Status|
|access to adequate healthcare|65   |93 |Access_To_Care  |
+-----------------------------+-----+---+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_sdoh_access_to_healthcare_wip|
|Compatibility:|Healthcare NLP 4.3.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|3.0 MB|

## Benchmarking

```bash
                 label	   tp	  fp	  fn	total	precision 	recall	      f1
Healthcare_Institution	 94.0	 8.0	 5.0	 99.0	 0.921569	0.949495	0.935323
        Access_To_Care	561.0	23.0	38.0	599.0	 0.960616	0.936561	0.948436
      Insurance_Status	 60.0	 5.0	 3.0	 63.0	 0.923077	0.952381	0.937500
```
