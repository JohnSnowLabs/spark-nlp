---
layout: model
title: Chunk Entity Resolver for billable ICD10-CM HCC codes
author: John Snow Labs
name: chunkresolve_icd10cm_hcc_clinical
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

This model maps extracted medical entities to ICD10-CM codes using chunk embeddings (augmented with synonyms, four times richer than previous resolver). It also adds support of 7-digit codes with HCC status.

For reference: http://www.formativhealth.com/wp-content/uploads/2018/06/HCC-White-Paper.pdf

## Predicted Entities

Outputs 7-digit billable ICD codes. In the result, look for `aux_label` parameter in the metadata to get HCC status. The HCC status can be divided to get further information: `billable status`, `hcc status`, and `hcc score`.

For example, in the example shared below the `billable status is 1`, `hcc status is 1`, and `hcc score is 8`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_hcc_clinical_en_3.0.0_3.0_1617356679231.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

...
resolver = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_hcc_clinical","en","clinical/models")    .setInputCols("token","chunk_embeddings")    .setOutputCol("entity")

pipeline = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk_embeddings, resolver])

data = spark.createDataFrame([["metastatic lung cancer"]]).toDF("text")
model = pipeline.fit(data)
results = model.transform(data)
...

```
```scala

...
val resolver = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_hcc_clinical","en","clinical/models")    .setInputCols("token","chunk_embeddings")    .setOutputCol("entity")

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
|Model Name:|chunkresolve_icd10cm_hcc_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, chunk_embeddings]|
|Output Labels:|[icd10cm_code]|
|Language:|en|
