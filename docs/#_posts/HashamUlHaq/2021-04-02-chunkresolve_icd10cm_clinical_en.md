---
layout: model
title: ICD10CM Entity Resolver
author: John Snow Labs
name: chunkresolve_icd10cm_clinical
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_clinical_en_3.0.0_3.0_1617355382919.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
icd10cm_resolution = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_clinical", "en", "clinical/models") \
  .setInputCols(["ner_token", "chunk_embeddings"]) \
  .setOutputCol("icd10cm_code") \
  .setDistanceFunction("COSINE")  \
  .setNeighbours(5)

pipeline_icd10cm = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, chunk_tokenizer, icd10cm_resolution])

pipeline_model = pipeline_icd10cm.fit(spark.createDataFrame([["""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection. She was on metformin, glipizide, and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG."""]]).toDF("text"))

result = pipeline_model.transform(data)
```
```scala
...
val icd10cm_resolution = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_clinical", "en", "clinical/models")
  .setInputCols("ner_token", "chunk_embeddings")
  .setOutputCol("icd10cm_code")
  .setDistanceFunction("COSINE")  
  .setNeighbours(5)

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, chunk_tokenizer, icd10cm_resolution))

val data = Seq("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection. She was on metformin, glipizide, and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|   | chunk                       | entity    | resolved_text                                      | code   | cms                                               |
|---|-----------------------------|-----------|----------------------------------------------------|--------|---------------------------------------------------|
| 0 | T2DM),                      | PROBLEM   | Type 2 diabetes mellitus with diabetic nephrop...  | E1121  | Type 2 diabetes mellitus with diabetic nephrop... |
| 1 | T2DM                        | PROBLEM   | Type 2 diabetes mellitus with diabetic nephrop...  | E1121  | Type 2 diabetes mellitus with diabetic nephrop... |
| 2 | polydipsia                  | PROBLEM   | Polydipsia                                         | R631   | Polydipsia:::Anhedonia:::Galactorrhea             |
| 3 | interference from turbidity | PROBLEM   | Non-working side interference                      | M2656  | Non-working side interference:::Hemoglobinuria... |
| 4 | polyuria                    | PROBLEM   | Other polyuria                                     | R358   | Other polyuria:::Polydipsia:::Generalized edem... |
| 5 | lipemia                     | PROBLEM   | Glycosuria                                         | R81    | Glycosuria:::Pure hyperglyceridemia:::Hyperchy... |
| 6 | starvation ketosis          | PROBLEM   | Propionic acidemia                                 | E71121 | Propionic acidemia:::Bartter's syndrome:::Hypo... |
| 7 | HTG                         | PROBLEM   | Pure hyperglyceridemia                             | E781   | Pure hyperglyceridemia:::Familial hypercholest... |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chunkresolve_icd10cm_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, chunk_embeddings]|
|Output Labels:|[icd10cm]|
|Language:|en|
