---
layout: model
title: ICD10CM Musculoskeletal Entity Resolver
author: John Snow Labs
name: chunkresolve_icd10cm_musculoskeletal_clinical
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

Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance

## Predicted Entities

ICD10-CM Codes and their normalized definition with `clinical_embeddings`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_musculoskeletal_clinical_en_3.0.0_3.0_1617355429847.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_musculoskeletal_clinical_en_3.0.0_3.0_1617355429847.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
muscu_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_musculoskeletal_clinical","en","clinical/models")\
	.setInputCols("token","chunk_embeddings")\
	.setOutputCol("entity")
pipeline_puerile = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, muscu_resolver])

model = pipeline_puerile.fit(spark.createDataFrame([["""The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. Mom states she had no fever. Her appetite was good but she was spitting up a lot. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion."""]]).toDF("text"))

results = model.transform(data)
```
```scala
...
val muscu_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_musculoskeletal_clinical","en","clinical/models")
	.setInputCols(Array("token","chunk_embeddings"))
	.setOutputCol("resolution")
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, muscu_resolver))

val data = Seq("The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. Mom states she had no fever. Her appetite was good but she was spitting up a lot. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
chunk     entity                            icd10_muscu_description  icd10_muscu_code

0         a cold, cough    PROBLEM  Postprocedural hemorrhage of a musculoskeletal...  M96831
1            runny nose    PROBLEM                         Acquired deformity of nose  M950
2                 fever    PROBLEM                           Periodic fever syndromes  M041
3  difficulty breathing    PROBLEM         Other dentofacial functional abnormalities  M2659
4             her cough    PROBLEM                                        Cervicalgia  M542
5         physical exam       TEST  Pathological fracture, unspecified toe(s), seq...  M84479S
6      fairly congested    PROBLEM  Synovial hypertrophy, not elsewhere classified...  M67262
7                Amoxil  TREATMENT                                        Torticollis  M436
8                 Aldex  TREATMENT  Other soft tissue disorders related to use, ov...  M7088
9  difficulty breathing    PROBLEM         Other dentofacial functional abnormalities  M2659
10       more congested    PROBLEM  Pain in unspecified ankle and joints of unspec...  M25579
11     trouble sleeping    PROBLEM                                      Low back pain  M545
12           congestion    PROBLEM                     Progressive systemic sclerosis  M340
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chunkresolve_icd10cm_musculoskeletal_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, chunk_embeddings]|
|Output Labels:|[icd10cm]|
|Language:|en|
