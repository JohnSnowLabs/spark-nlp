---
layout: model
title: Extract Income and Social Status Entities from Social Determinants of Health Texts
author: John Snow Labs
name: ner_sdoh_income_social_status_wip
date: 2023-02-10
tags: [licensed, clinical, social_determinants, en, ner, income, social_status, sdoh, public_health]
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

This model extracts income and social status information related to Social Determinants of Health from various kinds of biomedical documents.

## Predicted Entities

`Education`, `Marital_Status`, `Financial_Status`, `Population_Group`, `Employment`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_income_social_status_wip_en_4.2.8_3.0_1675999206708.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_income_social_status_wip_en_4.2.8_3.0_1675999206708.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = MedicalNerModel.pretrained("ner_sdoh_income_social_status_wip", "en", "clinical/models")\
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

sample_texts = ["Pt is described as divorced and pleasant when approached but keeps to himself. Pt is working as a plumber, but he gets financial diffuculties. He has a son student at college. His family is imigrant for 2 years."]

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

val ner_model = MedicalNerModel.pretrained("ner_sdoh_income_social_status_wip", "en", "clinical/models")
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

val data = Seq("Pt is described as divorced and pleasant when approached but keeps to himself. Pt is working as a plumber, but he gets financial diffuculties. He has a son student at college. His family is imigrant for 2 years.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------+----------------+-----+---+----------------------+
|sentence_id|ner_label       |begin|end|chunk                 |
+-----------+----------------+-----+---+----------------------+
|0          |Marital_Status  |19   |26 |divorced              |
|1          |Employment      |98   |104|plumber               |
|1          |Financial_Status|119  |140|financial diffuculties|
|2          |Education       |156  |162|student               |
|2          |Education       |167  |173|college               |
|3          |Population_Group|190  |197|imigrant              |
+-----------+----------------+-----+---+----------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_sdoh_income_social_status_wip|
|Compatibility:|Healthcare NLP 4.2.8+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|856.8 KB|

## Benchmarking

```bash
           label	    tp	   fp	    fn	 total	precision	   recall	       f1
       Education	  95.0	 20.0	  18.0	 113.0	 0.826087	 0.840708	 0.833333
Population_Group	  41.0	  0.0	   5.0	  46.0	 1.000000	 0.891304	 0.942529
Financial_Status	 286.0	 52.0	  82.0	 368.0	 0.846154	 0.777174	 0.810198
      Employment	3968.0	142.0	 215.0	4183.0	 0.965450	 0.948601	 0.956952
  Marital_Status	 167.0	  1.0	   7.0	 174.0	 0.994048	 0.959770	 0.976608
```