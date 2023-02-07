---
layout: model
title: Sentence Entity Resolver for ICD10-CM (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_icd10cm
language: en
repository: clinical/models
date: 2020-11-27
task: Entity Resolution
edition: Healthcare NLP 2.6.4
spark_version: 2.4
tags: [clinical,entity_resolution,en]
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
This model maps extracted medical entities to ICD10-CM codes using sentence embeddings.

{:.h2_title}
## Predicted Entities 
ICD10-CM Codes and their normalized definition with ``sbiobert_base_cased_mli`` sentence embeddings.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_en_2.6.4_2.4_1606235759310.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_en_2.6.4_2.4_1606235759310.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

icd10_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icd10_resolver])

data = spark.createDataFrame([["This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU ."]]).toDF("text")

results = nlpPipeline.fit(data).transform(data)

```
```scala
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli","en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

val icd10_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_icd10cm","en", "clinical/models")
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
+--------------------+-----+---+---------+------+----------+--------------------+--------------------+
|               chunk|begin|end|   entity|  code|confidence|   all_k_resolutions|         all_k_codes|
+--------------------+-----+---+---------+------+----------+--------------------+--------------------+
|        hypertension|   68| 79|  PROBLEM|  I150|    0.2606|Renovascular hype...|I150:::K766:::I10...|
|chronic renal ins...|   83|109|  PROBLEM|  N186|    0.2059|End stage renal d...|N186:::D631:::P96...|
|                COPD|  113|116|  PROBLEM| I2781|    0.2132|Cor pulmonale (ch...|I2781:::J449:::J4...|
|           gastritis|  120|128|  PROBLEM| K5281|    0.1425|Eosinophilic gast...|K5281:::K140:::K9...|
|                 TIA|  136|138|  PROBLEM|  G459|    0.1152|Transient cerebra...|G459:::I639:::T79...|
|a non-ST elevatio...|  182|202|  PROBLEM|  I214|    0.0889|Non-ST elevation ...|I214:::I256:::M62...|
|Guaiac positive s...|  208|229|  PROBLEM|  K626|    0.0631|Ulcer of anus and...|K626:::K380:::R15...|
|cardiac catheteri...|  295|317|     TEST|  Z950|    0.2549|Presence of cardi...|Z950:::Z95811:::I...|
|                PTCA|  324|327|TREATMENT| Z9861|    0.1268|Coronary angiopla...|Z9861:::Z9862:::I...|
|      mid LAD lesion|  332|345|  PROBLEM|L02424|    0.1117|Furuncle of left ...|L02424:::Q202:::L...|
+--------------------+-----+---+---------+------+----------+--------------------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:         | sbiobertresolve_icd10cm        |
| Type:          | SentenceEntityResolverModel     |
| Compatibility: | Spark NLP 2.6.4 +               |
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