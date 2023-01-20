---
layout: model
title: ICDO Entity Resolver
author: John Snow Labs
name: chunkresolve_icdo_clinical
date: 2021-04-02
tags: [entity_resolution, clinical, licensed, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
deprecated: true
annotator: ChunkEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance to map medical entities to ICD-O codes.

Given an oncological entity found in the text (via NER models like ner_jsl), it returns top terms and resolutions along with the corresponding `Morphology` codes comprising of `Histology` and `Behavior` codes.


## Predicted Entities

ICD-O Codes and their normalized definition with `clinical_embeddings`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICDO/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icdo_clinical_en_3.0.0_3.0_1617344918016.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icdo_clinical_en_3.0.0_3.0_1617344918016.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...

model = ChunkEntityResolverModel.pretrained("chunkresolve_icdo_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")

pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, clinical_ner_model, clinical_ner_chunker, chunk_embeddings, model])

data = ["""DIAGNOSIS: Left breast adenocarcinoma stage T3 N1b M0, stage IIIA.
She has been found more recently to have stage IV disease with metastatic deposits and recurrence involving the chest wall and lower left neck lymph nodes.
PHYSICAL EXAMINATION
NECK: On physical examination palpable lymphadenopathy is present in the left lower neck and supraclavicular area. No other cervical lymphadenopathy or supraclavicular lymphadenopathy is present.
RESPIRATORY: Good air entry bilaterally. Examination of the chest wall reveals a small lesion where the chest wall recurrence was resected. No lumps, bumps or evidence of disease involving the right breast is present.
ABDOMEN: Normal bowel sounds, no hepatomegaly. No tenderness on deep palpation. She has just started her last cycle of chemotherapy today, and she wishes to visit her daughter in Brooklyn, New York. After this she will return in approximately 3 to 4 weeks and begin her radiotherapy treatment at that time."""]

pipeline_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

light_pipeline = LightPipeline(pipeline_model)
result = light_pipeline.annotate(data)
```
```scala
...
val model = ChunkEntityResolverModel.pretrained("chunkresolve_icdo_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings, clinical_ner_model, clinical_ner_chunker, chunk_embeddings, model))

val data = Seq("DIAGNOSIS: Left breast adenocarcinoma stage T3 N1b M0, stage IIIA. She has been found more recently to have stage IV disease with metastatic deposits and recurrence involving the chest wall and lower left neck lymph nodes. PHYSICAL EXAMINATION NECK: On physical examination palpable lymphadenopathy is present in the left lower neck and supraclavicular area. No other cervical lymphadenopathy or supraclavicular lymphadenopathy is present. RESPIRATORY: Good air entry bilaterally. Examination of the chest wall reveals a small lesion where the chest wall recurrence was resected. No lumps, bumps or evidence of disease involving the right breast is present. ABDOMEN: Normal bowel sounds, no hepatomegaly. No tenderness on deep palpation. She has just started her last cycle of chemotherapy today, and she wishes to visit her daughter in Brooklyn, New York. After this she will return in approximately 3 to 4 weeks and begin her radiotherapy treatment at that time.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|   | chunk                      | begin | end | entity | idco_description                            | icdo_code |
|---|----------------------------|-------|-----|--------|---------------------------------------------|-----------|
| 0 | Left breast adenocarcinoma | 11    | 36  | Cancer | Intraductal carcinoma, noninfiltrating, NOS | 8500/2    |
| 1 | T3 N1b M0                  | 44    | 52  | Cancer | Kaposi sarcoma                              | 9140/3    |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chunkresolve_icdo_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, chunk_embeddings]|
|Output Labels:|[icd10pcs]|
|Language:|en|
