---
layout: model
title: Sentence Entity Resolver for billable ICD10-CM HCC codes
author: John Snow Labs
name: sbiobertresolve_icd10cm_augmented_billable_hcc
date: 2021-02-06
task: Entity Resolution
language: en
edition: Spark NLP 2.7.3
tags: [licensed, clinical, en, entity_resolution]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical entities to ICD10-CM codes using chunk embeddings (augmented with synonyms, four times richer than previous resolver). It also adds support of 7-digit codes with HCC status.

For reference: http://www.formativhealth.com/wp-content/uploads/2018/06/HCC-White-Paper.pdf

## Predicted Entities

Outputs 7-digit billable ICD codes. In the result, look for `aux_label` parameter in the metadata to get HCC status. The HCC status can be divided to get further information: `billable status`, `hcc status`, and `hcc score`.

For example, in the example shared `below the billable status is 1`, `hcc status is 1`, and `hcc score is 8`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_augmented_billable_hcc_en_2.7.3_2.4_1612609178670.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")
sbert_embedder = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["document"])\
     .setOutputCol("sbert_embeddings")
icd10_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_augmented_billable_hcc","en", "clinical/models") \
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
     .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["document"])\
     .setOutputCol("sbert_embeddings")
val icd10_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_augmented_billable_hcc","en", "clinical/models") \
     .setInputCols(["document", "sbert_embeddings"]) \
     .setOutputCol("icd10cm_code")\
     .setDistanceFunction("EUCLIDEAN").setReturnCosineDistances(True)
val bert_pipeline_icd = new Pipeline().setStages(Array(document_assembler, sbert_embedder, icd10_resolver))
val result = bert_pipeline_icd.fit(Seq.empty["metastatic lung cancer"].toDS.toDF("text")).transform(data)
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
|Model Name:|sbiobertresolve_icd10cm_augmented_billable_hcc|
|Compatibility:|Spark NLP 2.7.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[icd10cm_code]|
|Language:|en|