---
layout: model
title: Chunk Resolver (Cpt Clinical)
author: John Snow Labs
name: chunkresolve_cpt_clinical
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

Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance.

## Predicted Entities

CPT Codes and their normalized definition with `clinical_embeddings`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_cpt_clinical_en_3.0.0_3.0_1617355184583.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/chunkresolve_cpt_clinical_en_3.0.0_3.0_1617355184583.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
cpt_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_cpt_clinical","en","clinical/models")\
	.setInputCols("token","chunk_embeddings")\
	.setOutputCol("entity")
pipeline_puerile = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, cpt_resolver])

model = pipeline_puerile.fit(spark.createDataFrame([["""The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. Mom states she had no fever. Her appetite was good but she was spitting up a lot. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion."""]]).toDF("text"))

results = model.transform(data)
```
```scala
...
val cpt_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_cpt_clinical","en","clinical/models")
	.setInputCols(Array("token","chunk_embeddings"))
	.setOutputCol("resolution")
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, cpt_resolver))

val data = Seq("The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. Mom states she had no fever. Her appetite was good but she was spitting up a lot. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
chunk     entity                                     cpt_description  cpt_code

0         a cold, cough     PROBLEM  Thoracoscopy, surgical; with removal of a sing...  32669
1            runny nose     PROBLEM                         Unlisted procedure, larynx  31599
2                 fever     PROBLEM                            Cesarean delivery only;  59514
3  difficulty breathing     PROBLEM      Repair, laceration of diaphragm, any approach  39501
4            her cough      PROBLEM  Exploration for postoperative hemorrhage, thro...  35840
5         physical exam        TEST  Cesarean delivery only; including postpartum care  59515
6      fairly congested     PROBLEM               Pyelotomy; with drainage, pyelostomy  50125
7                Amoxil   TREATMENT      Cholecystoenterostomy; with gastroenterostomy  47721
8                 Aldex   TREATMENT  Laparoscopy, surgical; with omentopexy (omenta...  49326
9  difficulty breathing     PROBLEM      Repair, laceration of diaphragm, any approach  39501
10       more congested     PROBLEM            for section of 1 or more cranial nerves  61460
11     trouble sleeping     PROBLEM      Repair, laceration of diaphragm, any approach  39501
12           congestion     PROBLEM      Repair, laceration of diaphragm, any approach  39501
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chunkresolve_cpt_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, chunk_embeddings]|
|Output Labels:|[icd10pcs]|
|Language:|en|
