---
layout: model
title: Detect SDOH of Social Environment
author: John Snow Labs
name: ner_sdoh_social_environment_wip
date: 2023-02-10
tags: [licensed, clinical, social_determinants, en, ner, social, environment, sdoh, public_health]
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

This model extracts social environment terminologies related to Social Determinants of Health from various kinds of documents.

## Predicted Entities

`Social_Support`, `Chidhood_Event`, `Social_Exclusion`, `Violence_Abuse_Legal`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_social_environment_wip_en_4.2.8_3.0_1675998295035.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_social_environment_wip_en_4.2.8_3.0_1675998295035.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = MedicalNerModel.pretrained("ner_sdoh_social_environment_wip", "en", "clinical/models")\
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

sample_texts = ["He is the primary caregiver.",
             "There is some evidence of abuse.",
             "She stated that she was in a safe environment in prison, but that her siblings lived in an unsafe neighborhood, she was very afraid for them and witnessed their ostracism by other people.",
             "Medical history: Jane was born in a low - income household and experienced significant trauma during her childhood, including physical and emotional abuse."]

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

val ner_model = MedicalNerModel.pretrained("ner_sdoh_social_environment_wip", "en", "clinical/models")
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

val data = Seq("Medical history: Jane was born in a low - income household and experienced significant trauma during her childhood, including physical and emotional abuse.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

## Results

```bash
+--------------------+-----+---+---------------------------+
|ner_label           |begin|end|chunk                      |
+--------------------+-----+---+---------------------------+
|Social_Support      |10   |26 |primary caregiver          |
|Violence_Abuse_Legal|26   |30 |abuse                      |
|Violence_Abuse_Legal|49   |54 |prison                     |
|Social_Exclusion    |161  |169|ostracism                  |
|Chidhood_Event      |87   |113|trauma during her childhood|
|Violence_Abuse_Legal|139  |153|emotional abuse            |
+--------------------+-----+---+---------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_sdoh_social_environment_wip|
|Compatibility:|Healthcare NLP 4.2.8+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|858.7 KB|

## Benchmarking

```bash
	       label	    tp	  fp	   fn	  total	 precision	  recall	      f1
      Chidhood_Event	  34.0	 6.0	  5.0	   39.0	  0.850000	0.871795	0.860759
    Social_Exclusion	  45.0	 6.0	 12.0  	 57.0	  0.882353	0.789474	0.833333
      Social_Support	1139.0	57.0	103.0	 1242.0	  0.952341	0.917069	0.934372
Violence_Abuse_Legal	 235.0	38.0	 44.0	  279.0	  0.860806	0.842294	0.851449
```
