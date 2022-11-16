---
layout: model
title: Chunk Entity Resolver for ICD10 codes
author: John Snow Labs
name: chunkresolve_ICD10GM_2021
date: 2021-04-16
tags: [entity_resolution, clinical, licensed, de]
task: Entity Resolution
language: de
edition: Healthcare NLP 3.0.0
spark_version: 3.0
deprecated: true
annotator: ChunkEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical entities to ICD10-GM codes for German language using chunk embeddings (augmented with synonyms, four times richer than previous resolver).

## Predicted Entities

ICD10 codes

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_GM_DE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_ICD10GM_2021_de_3.0.0_3.0_1618603791008.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

...
resolver = ChunkEntityResolverModel.pretrained("chunkresolve_ICD10GM_2021","de","clinical/models")    .setInputCols("token","chunk_embeddings")    .setOutputCol("entity")

pipeline = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, resolver])

data = spark.createDataFrame([["metastatic lung cancer"]]).toDF("text")
model = pipeline.fit(data)
results = model.transform(data)
...

```
```scala

...
val resolver = ChunkEntityResolverModel.pretrained("chunkresolve_ICD10GM_2021","de","clinical/models")    .setInputCols("token","chunk_embeddings")    .setOutputCol("entity")

val pipeline = new Pipeline().setStages(Array(document_assembler, sbert_embedder, resolver))

val data = Seq("metastatic lung cancer").toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

## Results

```bash

|    | chunks                 | code   | resolutions                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | all_codes                                                                                              | billable_hcc_status_score   | all_distances                                                                                                            |
|---:|:-----------------------|:-------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------|:----------------------------|:-------------------------------------------------------------------------------------------------------------------------|
|  0 | metastatic lung cancer | C7800  | ['cancer metastatic to lung', 'metastasis from malignant tumor of lung', 'cancer metastatic to left lung', 'history of cancer metastatic to lung', 'metastatic cancer', 'history of cancer metastatic to lung (situation)', 'metastatic adenocarcinoma to bilateral lungs', 'cancer metastatic to chest wall', 'metastatic malignant neoplasm to left lower lobe of lung', 'metastatic carcinoid tumour', 'cancer metastatic to respiratory tract', 'metastatic carcinoid tumor'] | ['C7800', 'C349', 'C7801', 'Z858', 'C800', 'Z8511', 'C780', 'C798', 'C7802', 'C799', 'C7830', 'C7B00'] | ['1', '1', '8']             | ['0.0464', '0.0829', '0.0852', '0.0860', '0.0914', '0.0989', '0.1133', '0.1220', '0.1220', '0.1253', '0.1249', '0.1260'] |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chunkresolve_ICD10GM_2021|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, chunk_embeddings]|
|Output Labels:|[recognized]|
|Language:|de|
