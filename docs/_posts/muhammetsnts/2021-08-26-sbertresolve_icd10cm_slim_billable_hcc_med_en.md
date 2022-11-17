---
layout: model
title: Sentence Entity Resolver for Billable ICD10-CM HCC Codes (sbertresolve_icd10cm_slim_billable_hcc_med)
author: John Snow Labs
name: sbertresolve_icd10cm_slim_billable_hcc_med
date: 2021-08-26
tags: [icd10cm, entity_resolution, licensed, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.1.3
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts to ICD10 CM codes using sentence bert embeddings. In this model, synonyms having low cosine similarity to unnormalized terms are dropped. It also returns the official resolution text within the brackets inside the metadata. The model is augmented with synonyms, and previous augmentations are flexed according to cosine distances to unnormalized terms (ground truths).

## Predicted Entities

Outputs 7-digit billable ICD codes. In the result, look for aux_label parameter in the metadata to get HCC status. The HCC status can be divided to get further information: billable status, hcc status, and hcc score.For example, in the example shared below the billable status is 1, hcc status is 1, and hcc score is 11.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbertresolve_icd10cm_slim_billable_hcc_med_en_3.1.3_2.4_1629989198744.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings\
.pretrained('sbert_jsl_medium_uncased', 'en','clinical/models')\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

icd10_resolver = SentenceEntityResolverModel\
.pretrained("sbertresolve_icd10cm_slim_billable_hcc_med","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("icd10cm_code")\
.setDistanceFunction("EUCLIDEAN")\
.setReturnCosineDistances(True)

bert_pipeline_icd = Pipeline(stages = [document_assembler, sbert_embedder, icd10_resolver])

data = spark.createDataFrame([["bladder cancer"]]).toDF("text")

results = bert_pipeline_icd.fit(data).transform(data)
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbert_jsl_medium_uncased","en","clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sbert_embeddings")

val icd10_resolver = SentenceEntityResolverModel
.pretrained("sbertresolve_icd10cm_slim_billable_hcc_med","en", "clinical/models") 
.setInputCols(Array("ner_chunk", "sbert_embeddings")) 
.setOutputCol("icd10cm_code")
.setDistanceFunction("EUCLIDEAN")
.setReturnCosineDistances(True)

val bert_pipeline_icd = new Pipeline().setStages(Array(document_assembler, sbert_embedder, icd10_resolver))

val data = Seq("bladder cancer").toDF("text")

val result = bert_pipeline_icd.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.icd10cm.slim_billable_hcc_med").predict("""bladder cancer""")
```

</div>

## Results

```bash
|    | chunks         | code    | resolutions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | all_codes                                                                             | billable_hcc_status_score   | all_distances                                                                                                    |
|---:|:---------------|:--------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|--------------------------------------------------------------------------------------:|:----------------------------|:-----------------------------------------------------------------------------------------------------------------|
|  0 | bladder cancer | C671    |[bladder cancer, dome [Malignant neoplasm of dome of bladder], cancer of the urinary bladder [Malignant neoplasm of bladder, unspecified], prostate cancer [Malignant neoplasm of prostate], cancer of the urinary bladder, lateral wall [Malignant neoplasm of lateral wall of bladder], cancer of the urinary bladder, anterior wall [Malignant neoplasm of anterior wall of bladder], cancer of the urinary bladder, posterior wall [Malignant neoplasm of posterior wall of bladder], cancer of the urinary bladder, neck [Malignant neoplasm of bladder neck], cancer of the urinary bladder, ureteric orifice [Malignant neoplasm of ureteric orifice]]| [C671, C679, C61, C672, C673, C674, C675, C676, D090, Z126, D494, C670, Z8551, C7911] | ['1', '1', '11']            | [0.0894, 0.1051, 0.1184, 0.1180, 0.1200, 0.1204, 0.1255, 0.1375, 0.1357, 0.1452, 0.1469, 0.1513, 0.1500, 0.1575] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbertresolve_icd10cm_slim_billable_hcc_med|
|Compatibility:|Healthcare NLP 3.1.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk, sbert_embeddings]|
|Output Labels:|[icd10cm_code]|
|Language:|en|
|Case sensitive:|false|