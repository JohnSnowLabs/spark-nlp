---
layout: model
title: Sentence Entity Resolver for RxNorm (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_rxnorm
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
This model maps extracted medical entities to RxNorm codes using chunk embeddings.

{:.h2_title}
## Predicted Entities 
RxNorm Codes and their normalized definition with ``sbiobert_base_cased_mli`` embeddings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_en_2.6.4_2.4_1606235763316.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, rxnorm_resolver])

data = spark.createDataFrame([["This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU ."]]).toDF("text")

results = nlpPipeline.fit(data).transform(data)

```
```scala
...
val chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli","en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

val rxnorm_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_rxnorm","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, rxnorm_resolver))

val data = Seq("This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.h2_title}
## Results

```bash
+--------------------+-----+---+---------+-------+----------+-----------------------------------------------+--------------------+
|               chunk|begin|end|   entity|   code|confidence|                                    resolutions|               codes|
+--------------------+-----+---+---------+-------+----------+-----------------------------------------------+--------------------+
|        hypertension|   68| 79|  PROBLEM| 386165|    0.1567|hypercal:::hypersed:::hypertears:::hyperstat...|386165:::217667::...|
|chronic renal ins...|   83|109|  PROBLEM| 218689|    0.1036|nephro calci:::dialysis solutions:::creatini...|218689:::3310:::2...|
|                COPD|  113|116|  PROBLEM|1539999|    0.1644|broncomar dm:::acne medication:::carbon mono...|1539999:::214981:...|
|           gastritis|  120|128|  PROBLEM| 225965|    0.1983|gastroflux:::gastroflux oral product:::uceri...|225965:::1176661:...|
|                 TIA|  136|138|  PROBLEM|1089812|    0.0625|thera tears:::thiotepa injection:::nature's ...|1089812:::1660003...|
|a non-ST elevatio...|  182|202|  PROBLEM| 218767|    0.1007|non-aspirin pm:::aspirin-free:::non aspirin ...|218767:::215440::...|
|Guaiac positive s...|  208|229|  PROBLEM|1294361|    0.0820|anusol rectal product:::anusol hc rectal pro...|1294361:::1166715...|
|cardiac catheteri...|  295|317|     TEST| 385247|    0.1566|cardiacap:::cardiology pack:::cardizem:::car...|385247:::545063::...|
|                PTCA|  324|327|TREATMENT|   8410|    0.0867|alteplase:::reteplase:::pancuronium:::tripe ...|8410:::76895:::78...|
|      mid LAD lesion|  332|345|  PROBLEM| 151672|    0.0549|dulcolax:::lazerformalyde:::linaclotide:::du...|151672:::217985::...|
+--------------------+-----+---+---------+-------+----------+-----------------------------------------------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:         | sbiobertresolve_rxnorm        |
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
Trained on November 2020 RxNorm Clinical Drugs ontology graph with ``sbiobert_base_cased_mli`` embeddings.
https://www.nlm.nih.gov/pubs/techbull/nd20/brief/nd20_rx_norm_november_release.html