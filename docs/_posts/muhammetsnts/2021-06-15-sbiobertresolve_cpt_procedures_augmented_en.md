---
layout: model
title: Sentence Entity Resolver for CPT codes (Augmented)
author: John Snow Labs
name: sbiobertresolve_cpt_procedures_augmented
date: 2021-06-15
tags: [cpt, lincensed, en, clinical, licensed]
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

This model maps extracted medical entities to CPT codes using Sentence Bert Embeddings.

## Predicted Entities

CPT codes and their descriptions.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_cpt_procedures_augmented_en_3.1.0_3.0_1623789734339.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_cpt_procedures_augmented_en_3.1.0_3.0_1623789734339.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



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
.pretrained("sbiobertresolve_cpt_procedures_augmented", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("cpt_code")

cpt_pipelineModel = PipelineModel(
stages = [
documentAssembler,
jsl_sbert_embedder,
cpt_resolver])

cpt_lp = LightPipeline(cpt_pipelineModel)
result = cpt_lp.fullAnnotate("heart surgery")
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
.pretrained("sbiobertresolve_cpt_procedures_augmented", "en", "clinical/models") 
.setInputCols(Array("ner_chunk", "sbert_embeddings")) 
.setOutputCol("cpt_code")

val cpt_pipelineModel= new PipelineModel().setStages(Array(document_assembler, sbert_embedder, cpt_resolver))

val cpt_lp = LightPipeline(cpt_pipelineModel)
val result = cpt_lp.fullAnnotate("heart surgery")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.cpt.procedures_augmented").predict("""heart surgery""")
```

</div>

## Results

```bash
|    | chunks        | code  | resolutions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | all_codes                         | all_distances                         |
|---:|:--------------|:----- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------|:--------------------------------------|
|  0 | heart surgery | 33258 | [Cardiac surgery procedure [Operative tissue ablation and reconstruction of atria, performed at the time of other cardiac procedure(s), extensive (eg, maze procedure), without cardiopulmonary bypass (List separately in addition to code for primary procedure)], Cardiac surgery procedure [Unlisted procedure, cardiac surgery], Heart procedure [Interrogation device evaluation (in person) of intracardiac ischemia monitoring system with analysis, review, and report], Heart procedure [Insertion or removal and replacement of intracardiac ischemia monitoring system including imaging supervision and interpretation when performed and intra-operative interrogation and programming when performed; device only], ...]| [33258, 33999, 0306T, 0304T, ...] | [0.1031, 0.1031, 0.1377, 0.1377, ...] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_cpt_procedures_augmented|
|Compatibility:|Healthcare NLP 3.1.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[cpt_code]|
|Language:|en|
|Case sensitive:|true|

## Data Source

Trained on Current Procedural Terminology dataset with `sbiobert_base_cased_mli ` sentence embeddings.