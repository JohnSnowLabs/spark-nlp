---
layout: model
title: Sentence Entity Resolver for HCPCS Codes
author: John Snow Labs
name: sbiobertresolve_hcpcs
date: 2021-09-29
tags: [entity_resolution, hcpcs, licensed, en, clinical]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.2.3
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical entities to [Healthcare Common Procedure Coding System (HCPCS)](https://www.nlm.nih.gov/research/umls/sourcereleasedocs/current/HCPCS/index.html#:~:text=The%20Healthcare%20Common%20Procedure%20Coding,%2C%20supplies%2C%20products%20and%20services.)
codes using 'sbiobert_base_cased_mli' sentence embeddings. It also returns the domain information of the codes in the `all_k_aux_labels` parameter in the metadata of the result.

## Predicted Entities

`HCPCS Codes`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_HCPCS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_hcpcs_en_3.2.3_2.4_1632909577033.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

```sbiobertresolve_hcpcs``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_jsl``` as NER model. ```Procedure``` set in ```.setWhiteList()```.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings\
.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
.setInputCols(["ner_chunk"])\
.setOutputCol("sentence_embeddings")

hcpcs_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_hcpcs", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sentence_embeddings"]) \
.setOutputCol("hcpcs_code")\
.setDistanceFunction("EUCLIDEAN")

hcpcs_pipelineModel = PipelineModel(
stages = [
documentAssembler,
sbert_embedder,
hcpcs_resolver])

data = spark.createDataFrame([["Breast prosthesis, mastectomy bra, with integrated breast prosthesis form, unilateral, any size, any type"]]).toDF("text")

results = hcpcs_pipelineModel.fit(data).transform(data)
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sentence_embeddings")

val hcpcs_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_hcpcs", "en", "clinical/models") 
.setInputCols(Array("ner_chunk", "sentence_embeddings")) 
.setOutputCol("hcpcs_code")
.setDistanceFunction("EUCLIDEAN")

val hcpcs_pipeline = new Pipeline().setStages(Array(documentAssembler, sbert_embedder, hcpcs_resolver))

val data = Seq("Breast prosthesis, mastectomy bra, with integrated breast prosthesis form, unilateral, any size, any type").toDF("text")    

val results = hcpcs_pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.hcpcs").predict("""Breast prosthesis, mastectomy bra, with integrated breast prosthesis form, unilateral, any size, any type""")
```

</div>

## Results

```bash
+--+---------------------------------------------------------------------------------------------------------+----------+----------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+
|  |ner_chunk                                                                                                |hcpcs_code|all_codes                               |resolutions                                                                                                                                                                                                                                                                                                                                                                                                     |domain                                     |
+--+---------------------------------------------------------------------------------------------------------+----------+----------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+
|0 |Breast prosthesis, mastectomy bra, with integrated breast prosthesis form, unilateral, any size, any type|L8001     |[L8001, L8002, L8000, L8033, L8032, ...]|'Breast prosthesis, mastectomy bra, with integrated breast prosthesis form, unilateral, any size, any type', 'Breast prosthesis, mastectomy bra, with integrated breast prosthesis form, bilateral, any size, any type', 'Breast prosthesis, mastectomy bra, without integrated breast prosthesis form, any size, any type', 'Nipple prosthesis, custom fabricated, reusable, any material, any type, each', ...|Device, Device, Device, Device, Device, ...|
+--+---------------------------------------------------------------------------------------------------------+----------+----------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_hcpcs|
|Compatibility:|Healthcare NLP 3.2.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[hcpcs_code]|
|Language:|en|
|Case sensitive:|false|