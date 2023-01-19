---
layout: model
title: Sentence Entity Resolver for CPT codes (procedures and measurements) - Augmented
author: John Snow Labs
name: sbiobertresolve_cpt_procedures_measurements_augmented
date: 2021-07-02
tags: [licensed, en, entity_resolution, clinical]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.1.0
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps medical entities to CPT codes using Sentence Bert Embeddings. The corpus of this model has been extented to measurements, and this model is capable of mapping both procedures and measurement concepts/entities to CPT codes. Measurement codes are helpful in codifying medical entities related to tests and their results.

## Predicted Entities

CPT codes and their descriptions.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_cpt_procedures_measurements_augmented_en_3.1.0_3.0_1625257370771.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_cpt_procedures_measurements_augmented_en_3.1.0_3.0_1625257370771.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use


```sbiobertresolve_cpt_procedures_measurements_augmented``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_jsl``` as NER model. ```Procedure``` set in ```.setWhiteList()```.
```sbiobertresolve_cpt_procedures_measurements_augmented``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_measurements_clinical``` as NER model. ```Measurements``` set in ```.setWhiteList()```.
Merge ner_jsl and ner_measurements_clinical model chunks.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

jsl_sbert_embedder = BertSentenceEmbeddings\
.pretrained('sbiobert_base_cased_mli','en','clinical/models')\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

cpt_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_cpt_procedures_measurements_augmented", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("cpt_code")

cpt_pipelineModel = PipelineModel(
stages = [
documentAssembler,
jsl_sbert_embedder,
cpt_resolver])

cpt_lp = LightPipeline(cpt_pipelineModel)
result = cpt_lp.fullAnnotate(['calcium score', 'heart surgery'])
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli","en","clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sbert_embeddings")

val cpt_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_cpt_procedures_measurements_augmented", "en", "clinical/models) 
.setInputCols(Array("ner_chunk", "sbert_embeddings")) 
.setOutputCol("cpt_code")

val cpt_pipelineModel= new PipelineModel().setStages(Array(document_assembler, sbert_embedder, cpt_resolver))

val cpt_lp = LightPipeline(cpt_pipelineModel)
val result = cpt_lp.fullAnnotate(['calcium score', 'heart surgery'])
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.cpt.procedures_measurements").predict("""calcium score""")
```

</div>

## Results

```bash
|    | chunks        | code  | resolutions                                                                                                                                                                                                                               |
|---:|:--------------|:----- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | calcium score | 82310 | Calcium measurement [Calcium; total]                                                                                                                                                                                                      |
|  1 | heart surgery | 33257 | Cardiac surgery procedure [Operative tissue ablation and reconstruction of atria, performed at the time of other cardiac procedure(s), limited (eg, modified maze procedure) (List separately in addition to code for primary procedure)] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_cpt_procedures_measurements_augmented|
|Compatibility:|Healthcare NLP 3.1.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sbert_embeddings]|
|Output Labels:|[cpt_code]|
|Language:|en|
|Case sensitive:|true|

## Data Source

Trained on Current Procedural Terminology dataset with `sbiobert_base_cased_mli` sentence embeddings.