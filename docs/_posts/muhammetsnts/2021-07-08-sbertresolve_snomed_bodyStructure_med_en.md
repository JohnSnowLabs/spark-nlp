---
layout: model
title: Sentence Entity Resolver for Snomed Concepts, Body Structure Version
author: John Snow Labs
name: sbertresolve_snomed_bodyStructure_med
date: 2021-07-08
tags: [snomed, en, entity_resolution, licensed]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.1.0
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical (anatomical structures) entities to Snomed codes (body structure version) using sentence embeddings.

## Predicted Entities

Snomed Codes and their normalized definition with `sbert_jsl_medium_uncased ` embeddings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbertresolve_snomed_bodyStructure_med_en_3.1.0_2.4_1625772026635.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbertresolve_snomed_bodyStructure_med_en_3.1.0_2.4_1625772026635.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

jsl_sbert_embedder = BertSentenceEmbeddings\
.pretrained('sbert_jsl_medium_uncased','en','clinical/models')\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

snomed_resolver = SentenceEntityResolverModel\
.pretrained("sbertresolve_snomed_bodyStructure_med", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("snomed_code")

snomed_pipelineModel = PipelineModel(
stages = [
documentAssembler,
jsl_sbert_embedder,
snomed_resolver])

snomed_lp = LightPipeline(snomed_pipelineModel)
result = snomed_lp.fullAnnotate("Amputation stump")
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbert_jsl_medium_uncased","en","clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sbert_embeddings")

val snomed_resolver = SentenceEntityResolverModel
.pretrained("sbertresolve_snomed_bodyStructure_med", "en", "clinical/models") 
.setInputCols(Array("ner_chunk", "sbert_embeddings")) 
.setOutputCol("snomed_code")

val snomed_pipelineModel= new PipelineModel().setStages(Array(document_assembler, sbert_embedder, snomed_resolver))

val snomed_lp = LightPipeline(snomed_pipelineModel)
val result = snomed_lp.fullAnnotate("Amputation stump")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.snomed_body_structure_med").predict("""Amputation stump""")
```

</div>

## Results

```bash
|    | chunks           | code     | resolutions                                                                                                                                                                                                                                  | all_codes                                                                                       | all_distances                                                               |
|---:|:-----------------|:---------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------|
|  0 | amputation stump | 38033009 | [Amputation stump, Amputation stump of upper limb, Amputation stump of left upper limb, Amputation stump of lower limb, Amputation stump of left lower limb, Amputation stump of right upper limb, Amputation stump of right lower limb, ...]| ['38033009', '771359009', '771364008', '771358001', '771367001', '771365009', '771368006', ...] | ['0.0000', '0.0773', '0.0858', '0.0863', '0.0905', '0.0911', '0.0972', ...] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbertresolve_snomed_bodyStructure_med|
|Compatibility:|Healthcare NLP 3.1.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[snomed_code]|
|Language:|en|
|Case sensitive:|true|

## Data Source

https://www.snomed.org/