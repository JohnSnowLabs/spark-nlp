---
layout: model
title: Extract Community Condition Entities from Social Determinants of Health Texts
author: John Snow Labs
name: ner_sdoh_community_condition_wip
date: 2023-02-24
tags: [licensed, en, clinical, sdoh, social_determinants, ner, public_health, community, condition, community_condition]
task: Named Entity Recognition
language: en
nav_key: models
edition: Healthcare NLP 4.3.1
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts community condition information related to Social Determinants of Health from various kinds of biomedical documents.

## Predicted Entities

`Transportation`, `Community_Living_Conditions`, `Housing`, `Food_Insecurity`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_community_condition_wip_en_4.3.1_3.0_1677201525944.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_community_condition_wip_en_4.3.1_3.0_1677201525944.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = MedicalNerModel.pretrained("ner_sdoh_community_condition_wip", "en", "clinical/models")\
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

sample_texts = ["He is currently experiencing financial stress due to job insecurity, and he lives in a small apartment in a densely populated area with limited access to green spaces and outdoor recreational activities.",
             "Patient reports difficulty affording healthy food, and relies oncheaper, processed options.",
               "She reports her husband and sons provide transportation top medical apptsand do her grocery shopping."]


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

val ner_model = MedicalNerModel.pretrained("ner_sdoh_community_condition_wip", "en", "clinical/models")
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

val data = Seq("He is currently experiencing financial stress due to job insecurity, and he lives in a small apartment in a densely populated area with limited access to green spaces and outdoor recreational activities.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------------------------+-----+---+---------------------------+
|chunk                          |begin|end|ner_label                  |
+-------------------------------+-----+---+---------------------------+
|small apartment                |87   |101|Housing                    |
|green spaces                   |154  |165|Community_Living_Conditions|
|outdoor recreational activities|171  |201|Community_Living_Conditions|
|healthy food                   |37   |48 |Food_Insecurity            |
|transportation                 |41   |54 |Transportation             |
+-------------------------------+-----+---+---------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_sdoh_community_condition_wip|
|Compatibility:|Healthcare NLP 4.3.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|3.0 MB|

## Benchmarking

```bash
                      label 	 tp	  fp	  fn	total	precision	  recall	      f1
            Food_Insecurity	 40.0	 0.0	 5.0	 45.0	 1.000000	0.888889	0.941176
                    Housing	376.0	20.0	28.0	404.0	 0.949495	0.930693	0.940000
Community_Living_Conditions	 97.0	 8.0	 8.0	105.0	 0.923810	0.923810	0.923810
             Transportation	 31.0	 2.0	 0.0	 31.0	 0.939394	1.000000	0.968750
```
