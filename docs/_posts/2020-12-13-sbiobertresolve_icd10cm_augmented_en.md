---
layout: model
title: Sentence Entity Resolver for ICD10-CM (Augmented)
author: John Snow Labs
name: sbiobertresolve_icd10cm_augmented
language: en
repository: clinical/models
date: 2020-12-13
task: Entity Resolution
edition: Healthcare NLP 2.6.5
spark_version: 2.4
tags: [clinical,entity_resolution,en]
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
This model maps extracted medical entities to ICD10-CM codes using chunk embeddings (augmented with synonyms, four times richer than previous resolver).

{:.h2_title}
## Predicted Entities 
ICD10-CM Codes and their normalized definition with ``sbiobert_base_cased_mli`` sentence embeddings.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_augmented_en_2.6.4_2.4_1607890300949.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_augmented_en_2.6.4_2.4_1607890300949.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

icd10_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_augmented","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icd10_resolver])

data = spark.createDataFrame([["This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU ."]]).toDF("text")

results = nlpPipeline.fit(data).transform(data)

```
```scala
...
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli","en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

val icd10_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_icd10cm_augmented","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icd10_resolver))

val data = Seq("This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.h2_title}
## Results

```bash
+--------------------+---------+------+------------------------------------------+---------------------+
|               chunk|   entity|  code|                         all_k_resolutions|          all_k_codes|
+--------------------+---------+------+------------------------------------------+---------------------+
|        hypertension|  PROBLEM|   I10|hypertension:::hypertension monitored::...|I10:::Z8679:::I159...|
|chronic renal ins...|  PROBLEM|  N189|chronic renal insufficiency:::chronic r...|N189:::P2930:::N19...|
|                COPD|  PROBLEM|  J449|copd - chronic obstructive pulmonary di...|J449:::J984:::J628...|
|           gastritis|  PROBLEM| K2970|gastritis:::chemical gastritis:::gastri...|K2970:::K2960:::K2...|
|                 TIA|  PROBLEM| S0690|cerebral trauma (disorder):::cerebral c...|S0690:::S060X:::G4...|
|a non-ST elevatio...|  PROBLEM|  I219|silent myocardial infarction (disorder)...|I219:::I248:::I256...|
|Guaiac positive s...|  PROBLEM|  K921|guaiac-positive stools:::acholic stool ...|K921:::R195:::R15:...|
|      mid LAD lesion|  PROBLEM| I2102|stemi involving left anterior descendin...|I2102:::I2101:::Q2...|
+--------------------+---------+------+------------------------------------------+---------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:         | sbiobertresolve_icd10cm_augmented|
| Type:          | SentenceEntityResolverModel     |
| Compatibility: | Spark NLP 2.6.5 +               |
| License:       | Licensed            |
| Edition:       | Official          |
|Input labels:        | [ner_chunk, chunk_embeddings]     |
|Output labels:       | [resolution]                 |
| Language:      | en                  |
| Dependencies: | sbiobert_base_cased_mli |

{:.h2_title}
## Data Source
Trained on ICD10 Clinical Modification dataset with ``sbiobert_base_cased_mli`` sentence embeddings.
https://www.icd10data.com/ICD10CM/Codes/