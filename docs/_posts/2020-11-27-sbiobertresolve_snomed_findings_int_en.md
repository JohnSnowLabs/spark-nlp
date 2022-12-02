---
layout: model
title: Sentence Entity Resolver for Snomed Concepts, INT version (``sbiobert_base_cased_mli`` embeddings)
author: John Snow Labs
name: sbiobertresolve_snomed_findings_int
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
This model maps extracted medical entities to Snomed codes (INT version) using chunk embeddings.

{:.h2_title}
## Predicted Entities 
Snomed Codes and their normalized definition with ``sbiobert_base_cased_mli`` embeddings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_snomed_findings_int_en_2.6.4_2.4_1606235761314.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## Predicted Entities 
Snomed Codes and their normalized definition with ``sbiobert_base_cased_mli`` embeddings.

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

snomed_int_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_snomed_findings_int","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, snomed_int_resolver])

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

val snomed_int_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_snomed_findings_int","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, snomed_int_resolver))

val data = Seq("This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.h2_title}
## Results

```bash
+--------------------+-----+---+---------+---------------+----------+--------------------+--------------------+
|               chunk|begin|end|   entity|           code|confidence|         resolutions|               codes|
+--------------------+-----+---+---------+---------------+----------+--------------------+--------------------+
|        hypertension|   68| 79|  PROBLEM|      266285003|    0.8867|rheumatic myocard...|266285003:::15529...|
|chronic renal ins...|   83|109|  PROBLEM|      236425005|    0.2470|chronic renal imp...|236425005:::90688...|
|                COPD|  113|116|  PROBLEM|      413839001|    0.0720|chronic lung dise...|413839001:::41384...|
|           gastritis|  120|128|  PROBLEM|      266502003|    0.3240|acute peptic ulce...|266502003:::45560...|
|                 TIA|  136|138|  PROBLEM|353101000119105|    0.0727|prostatic intraep...|353101000119105::...|
|a non-ST elevatio...|  182|202|  PROBLEM|      233843008|    0.2846|silent myocardial...|233843008:::71942...|
|Guaiac positive s...|  208|229|  PROBLEM|      168319009|    0.1167|stool culture pos...|168319009:::70396...|
|cardiac catheteri...|  295|317|     TEST|      301095005|    0.2137|cardiac finding::...|301095005:::25090...|
|                PTCA|  324|327|TREATMENT|842741000000109|    0.0631|occlusion of post...|842741000000109::...|
|      mid LAD lesion|  332|345|  PROBLEM|      449567000|    0.0808|overriding left v...|449567000:::25342...|
+--------------------+-----+---+---------+---------------+----------+--------------------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:         | sbiobertresolve_snomed_findings_int        |
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