---
layout: model
title: Sentence Entity Resolver for Billable ICD10-CM HCC Codes (slim)
author: John Snow Labs
name: sbiobertresolve_icd10cm_slim_billable_hcc
date: 2021-05-25
tags: [icd10cm, slim, licensed, en]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 3.0.3
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts to ICD10 CM codes using sentence biobert embeddings. In this model, synonyms having low cosine similarity to unnormalized terms are dropped. It also returns the official resolution text within the brackets inside the metadata. The model is augmented with synonyms, and previous augmentations are flexed according to cosine distances to unnormalized terms (ground truths).

## Predicted Entities

Outputs 7-digit billable ICD codes. In the result, look for aux_label parameter in the metadata to get HCC status. The HCC status can be divided to get further information: billable status, hcc status, and hcc score.For example, in the example shared below the billable status is 1, hcc status is 1, and hcc score is 11.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_slim_billable_hcc_en_3.0.3_3.0_1621942329774.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")
sbert_embedder = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
     .setInputCols(["document"])\
     .setOutputCol("sbert_embeddings")
icd10_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_slim_billable_hcc ","en", "clinical/models") \
     .setInputCols(["document", "sbert_embeddings"]) \
     .setOutputCol("icd10cm_code")\
     .setDistanceFunction("EUCLIDEAN").setReturnCosineDistances(True)
bert_pipeline_icd = PipelineModel(stages = [document_assembler, sbert_embedder, icd10_resolver])

model = bert_pipeline_icd.fit(spark.createDataFrame([["metastatic lung cancer"]]).toDF("text"))
results = model.transform(data)
```
```scala
val document_assembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")
val sbert_embedder = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
     .setInputCols(["document"])\
     .setOutputCol("sbert_embeddings")
val icd10_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_slim_billable_hcc","en", "clinical/models") \
     .setInputCols(["document", "sbert_embeddings"]) \
     .setOutputCol("icd10cm_code")\
     .setDistanceFunction("EUCLIDEAN").setReturnCosineDistances(True)
val bert_pipeline_icd = new Pipeline().setStages(Array(document_assembler, sbert_embedder, icd10_resolver))
val result = bert_pipeline_icd.fit(Seq.empty["metastatic lung cancer"].toDS.toDF("text")).transform(data)
```
</div>

## Results

```bash
|    | chunks         | code    | resolutions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | all_codes                                  | billable_hcc_status_score   | all_distances                                            |
|---:|:---------------|:--------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|-------------------------------------------:|:----------------------------|:---------------------------------------------------------|
|  0 | bladder cancer | C671    |[bladder cancer, dome [Malignant neoplasm of dome of bladder], cancer of the urinary bladder [Malignant neoplasm of bladder, unspecified], adenocarcinoma, bladder neck [Malignant neoplasm of bladder neck], cancer in situ of urinary bladder [Carcinoma in situ of bladder], cancer of the urinary bladder, ureteric orifice [Malignant neoplasm of ureteric orifice], tumor of bladder neck [Neoplasm of unspecified behavior of bladder], cancer of the urethra [Malignant neoplasm of urethra]]| [C671, C679, C675, D090, C676, D494, C680] | ['1', '1', '11']            | [0.0685, 0.0709, 0.0963, 0.0978, 0.1068, 0.1080, 0.1211] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_icd10cm_slim_billable_hcc|
|Compatibility:|Spark NLP for Healthcare 3.0.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[icd10cm_code]|
|Language:|en|
|Case sensitive:|false|