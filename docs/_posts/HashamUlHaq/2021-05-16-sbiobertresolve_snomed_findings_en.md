---
layout: model
title: Sentence Entity Resolver for Snomed Concepts, CT version (``sbiobert_base_cased_mli`` embeddings)
author: John Snow Labs
name: sbiobertresolve_snomed_findings
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

This model maps extracted medical entities to Snomed codes (CT version) using `sbiobert_base_cased_mli` Sentence Bert Embeddings, and has faster load time, with a speedup of about 6X when compared to previous versions. Also the load process now is more memory friendly meaning that the maximum memory required during load time is smaller, reducing the chances of OOM exceptions, and thus relaxing hardware requirements.

## Predicted Entities

Predicts Snomed Codes and their normalized definition for each chunk.

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_snomed_findings_en_3.0.4_3.0_1621191323188.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_snomed_findings_en_3.0.4_3.0_1621191323188.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

```sbiobertresolve_snomed_findings``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_clinical``` as NER model. No need to set ```.setWhiteList()```.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

snomed_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_snomed_findings","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, snomed_resolver])

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

val snomed_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_snomed_findings","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, snomed_resolver))

val data = Seq("This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.snomed.findings").predict("""This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .""")
```

</div>

## Results

```bash
+--------------------+-----+---+---------+---------+----------+--------------------+--------------------+
|               chunk|begin|end|   entity|     code|confidence|         resolutions|               codes|
+--------------------+-----+---+---------+---------+----------+--------------------+--------------------+
|        hypertension|   68| 79|  PROBLEM| 38341003|    0.3234|hypertension:::hy...|38341003:::155295...|
|chronic renal ins...|   83|109|  PROBLEM|723190009|    0.7522|chronic renal ins...|723190009:::70904...|
|                COPD|  113|116|  PROBLEM| 13645005|    0.1226|copd - chronic ob...|13645005:::155565...|
|           gastritis|  120|128|  PROBLEM|235653009|    0.2444|gastritis:::gastr...|235653009:::45560...|
|                 TIA|  136|138|  PROBLEM|275382005|    0.0766|cerebral trauma (...|275382005:::44739...|
|a non-ST elevatio...|  182|202|  PROBLEM|233843008|    0.2224|silent myocardial...|233843008:::19479...|
|Guaiac positive s...|  208|229|  PROBLEM| 59614000|    0.9678|guaiac-positive s...|59614000:::703960...|
|cardiac catheteri...|  295|317|     TEST|301095005|    0.2584|cardiac finding::...|301095005:::25090...|
|                PTCA|  324|327|TREATMENT|373108000|    0.0809|post percutaneous...|373108000:::25103...|
|      mid LAD lesion|  332|345|  PROBLEM|449567000|    0.0900|overriding left v...|449567000:::46140...|
+--------------------+-----+---+---------+---------+----------+--------------------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_snomed_findings|
|Compatibility:|Healthcare NLP 3.0.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk, sbert_embeddings]|
|Output Labels:|[snomed_ct_code]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on SNOMED (CT version) Findings with ``sbiobert_base_cased_mli`` sentence embeddings.
http://www.snomed.org/