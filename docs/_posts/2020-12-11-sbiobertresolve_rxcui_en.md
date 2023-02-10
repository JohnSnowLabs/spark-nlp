---
layout: model
title: Sentence Entity Resolver for RxCUI (``sbiobert_base_cased_mli`` embeddings)
author: John Snow Labs
name: sbiobertresolve_rxcui
language: en
repository: clinical/models
date: 2020-12-11
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
This model maps extracted medical entities to RxCUI codes using chunk embeddings.

{:.h2_title}
## Predicted Entities 
RxCUI Codes and their normalized definition with ``sbiobert_base_cased_mli`` embeddings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxcui_en_2.6.4_2.4_1607714146277.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxcui_en_2.6.4_2.4_1607714146277.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use 

```sbiobertresolve_rxcui``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_posology``` as NER model. ```DRUG``` set in ```.setWhiteList()```.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

rxcui_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_rxcui","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, rxcui_resolver])

data = spark.createDataFrame([["He was seen by the endocrinology service and she was discharged on 50 mg of eltrombopag oral at night, 5 mg amlodipine with meals, and metformin 1000 mg two times a day"]]).toDF("text")

results = nlpPipeline.fit(data).transform(data)

```
```scala
...
val chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli","en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

val rxcui_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_rxcui","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, rxcui_resolver))

val data = Seq("He was seen by the endocrinology service and she was discharged on 50 mg of eltrombopag oral at night, 5 mg amlodipine with meals, and metformin 1000 mg two times a day").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.h2_title}
## Results

```bash
+---------------------------+--------+-----------------------------------------------------+
| chunk                     | code   | term                                                |               
+---------------------------+--------+-----------------------------------------------------+
| 50 mg of eltrombopag oral | 825427 | eltrombopag 50 MG Oral Tablet                       |
| 5 mg amlodipine           | 197361 | amlodipine 5 MG Oral Tablet                         |
| metformin 1000 mg         | 861004 | metformin hydrochloride 2000 MG Oral Tablet         |
+---------------------------+--------+-----------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:         | sbiobertresolve_rxcui         |
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
Trained on November 2020 RxNorm Clinical Drugs ontology graph with ``sbiobert_base_cased_mli`` embeddings.
https://www.nlm.nih.gov/pubs/techbull/nd20/brief/nd20_rx_norm_november_release.html.
[Sample Content](https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui=1000000).