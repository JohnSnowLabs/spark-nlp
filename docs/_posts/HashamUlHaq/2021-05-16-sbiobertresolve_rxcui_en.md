---
layout: model
title: Sentence Entity Resolver for RxCUI (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_rxcui
date: 2021-05-16
tags: [entity_resolution, clinical, licensed, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.0.4
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts (like drugs/ingredients) to RxCUI codes codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings, and has faster load time, with a speedup of about 6X when compared to previous versions. Also the load process now is more memory friendly meaning that the maximum memory required during load time is smaller, reducing the chances of OOM exceptions, and thus relaxing hardware requirements.

## Predicted Entities

Predicts RxCUI Codes and their normalized definition for each chunk.

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxcui_en_3.0.4_3.0_1621189488426.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxcui_en_3.0.4_3.0_1621189488426.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

rxcui_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxcui","en", "clinical/models") \
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

val rxcui_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_rxcui","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, rxcui_resolver))

val data = Seq("He was seen by the endocrinology service and she was discharged on 50 mg of eltrombopag oral at night, 5 mg amlodipine with meals, and metformin 1000 mg two times a day").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.rxcui").predict("""He was seen by the endocrinology service and she was discharged on 50 mg of eltrombopag oral at night, 5 mg amlodipine with meals, and metformin 1000 mg two times a day""")
```

</div>

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
|---|---|
|Model Name:|sbiobertresolve_rxcui|
|Compatibility:|Healthcare NLP 3.0.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxcui_code]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on November 2020 RxNorm Clinical Drugs ontology graph with ``sbiobert_base_cased_mli`` embeddings.
https://www.nlm.nih.gov/pubs/techbull/nd20/brief/nd20_rx_norm_november_release.html.
[Sample Content](https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui=1000000).