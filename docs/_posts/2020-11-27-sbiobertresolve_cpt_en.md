---
layout: model
title: Sentence Entity Resolver for CPT (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_cpt
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
This model maps extracted medical entities to CPT codes using chunk embeddings.

{:.h2_title}
## Predicted Entities 
CPT Codes and their normalized definition with ``sbiobert_base_cased_mli`` sentence embeddings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_cpt_en_2.6.4_2.4_1606235767322.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

cpt_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_cpt","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, cpt_resolver])

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

val cpt_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_cpt","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, cpt_resolver))

val data = Seq("This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.h2_title}
## Results

```bash
+--------------------+-----+---+---------+-----+----------+--------------------+--------------------+
|               chunk|begin|end|   entity| code|confidence|   all_k_resolutions|         all_k_codes|
+--------------------+-----+---+---------+-----+----------+--------------------+--------------------+
|        hypertension|   68| 79|  PROBLEM|49425|    0.0967|Insertion of peri...|49425:::36818:::3...|
|chronic renal ins...|   83|109|  PROBLEM|50070|    0.2569|Nephrolithotomy; ...|50070:::49425:::5...|
|                COPD|  113|116|  PROBLEM|49425|    0.0779|Insertion of peri...|49425:::31592:::4...|
|           gastritis|  120|128|  PROBLEM|43810|    0.5289|Gastroduodenostom...|43810:::43880:::4...|
|                 TIA|  136|138|  PROBLEM|25927|    0.2060|Transmetacarpal a...|25927:::25931:::6...|
|a non-ST elevatio...|  182|202|  PROBLEM|33300|    0.3046|Repair of cardiac...|33300:::33813:::3...|
|Guaiac positive s...|  208|229|  PROBLEM|47765|    0.0974|Anastomosis, of i...|47765:::49425:::1...|
|cardiac catheteri...|  295|317|     TEST|62225|    0.1996|Replacement or ir...|62225:::33722:::4...|
|                PTCA|  324|327|TREATMENT|60500|    0.1481|Parathyroidectomy...|60500:::43800:::2...|
|      mid LAD lesion|  332|345|  PROBLEM|33722|    0.3097|Closure of aortic...|33722:::33732:::3...|
+--------------------+-----+---+---------+-----+----------+--------------------+--------------------+
```
{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:         | sbiobertresolve_cpt        |
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
Trained on Current Procedural Terminology dataset with ``sbiobert_base_cased_mli`` sentence embeddings.