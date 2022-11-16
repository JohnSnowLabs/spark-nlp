---
layout: model
title: Sentence Entity Resolver for Snomed Aux Concepts, INT version (``sbiobert_base_cased_mli`` embeddings)
author: John Snow Labs
name: sbiobertresolve_snomed_auxConcepts_int
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
This model maps extracted medical entities to Snomed codes (with Morph Abnormality, Procedure, Substance, Physical Object, Body Structure concepts from INT version) using chunk embeddings.

{:.h2_title}
## Predicted Entities 
Snomed Codes and their normalized definition with ``sbiobert_base_cased_mli`` embeddings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_snomed_auxConcepts_int_en_2.6.4_2.4_1606235764318.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

snomed_aux_int_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_auxConcepts_int","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, snomed_aux_int_resolver])

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

val snomed_aux_int_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_snomed_auxConcepts_int","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, snomed_aux_int_resolver))

val data = Seq("This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.h2_title}
## Results

```bash
+--------------------+-----+---+---------+---------------+----------+--------------------+--------------------+
|               chunk|begin|end|   entity|           code|confidence|         resolutions|               codes|
+--------------------+-----+---+---------+---------------+----------+--------------------+--------------------+
|        hypertension|   68| 79|  PROBLEM|      148439002|    0.2138|risk factors pres...|148439002:::42595...|
|chronic renal ins...|   83|109|  PROBLEM|      722403003|    0.8517|gastrointestinal ...|722403003:::13781...|
|                COPD|  113|116|  PROBLEM|845101000000100|    0.0962|management of chr...|845101000000100::...|
|           gastritis|  120|128|  PROBLEM|      711498001|    0.3398|magnetic resonanc...|711498001:::71771...|
|                 TIA|  136|138|  PROBLEM|      449758002|    0.1927|traumatic infarct...|449758002:::85844...|
|a non-ST elevatio...|  182|202|  PROBLEM|  1411000087101|    0.0823|ct of left knee::...|1411000087101:::3...|
|Guaiac positive s...|  208|229|  PROBLEM|      388507006|    0.0555|asparagus rast:::...|388507006:::71771...|
|cardiac catheteri...|  295|317|     TEST|       41976001|    0.9790|cardiac catheteri...|41976001:::705921...|
|                PTCA|  324|327|TREATMENT|      312644004|    0.0616|angioplasty of po...|312644004:::41507...|
|      mid LAD lesion|  332|345|  PROBLEM|       91749005|    0.1399|structure of firs...|91749005:::917470...|
+--------------------+-----+---+---------+---------------+----------+--------------------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:         | sbiobertresolve_snomed_auxConcepts_int        |
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
Trained on SNOMED (INT version) Findings with ``sbiobert_base_cased_mli`` sentence embeddings.
http://www.snomed.org/