---
layout: model
title: Sentence Entity Resolver for Snomed Concepts
author: John Snow Labs
name: sbiobertresolve_snomed_findings_aux_concepts
date: 2021-07-14
tags: [snomed, licensed, en, clinical]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.1.2
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts to Snomed codes using sbiobert_base_cased_mli Sentence Bert Embeddings. This is also capable of extracting Morph Abnormality, Procedure, Substance, Physical Object, and Body Structure concepts of Snomed codes.

## Predicted Entities

Predicts Snomed Codes and their normalized definition for each chunk. In the metadata, the `all_k_aux_labels` can be divided to get further information: `ground truth`, `concept`, and `aux` .For example, in the example shared below the ground truth is `Atherosclerosis`, concept is `Observation`, and aux is `Morph Abnormality`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_snomed_findings_aux_concepts_en_3.1.2_3.0_1626280542687.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings\
     .pretrained('sbiobert_base_cased_mli','en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sbert_embeddings")

snomed_resolver = SentenceEntityResolverModel\
     .pretrained("sbiobertresolve_snomed_findings_aux_concepts", "en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("snomed_code")\
     .setDistanceFunction("EUCLIDEAN")

snomed_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        snomed_resolver])

snomed_lp = LightPipeline(snomed_pipelineModel)
result = snomed_lp.fullAnnotate("atherosclerosis")
```
```scala
val document_assembler = DocumentAssembler()
     .setInputCol("text")
     .setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
     .pretrained("sbiobert_base_cased_mli","en","clinical/models")
     .setInputCols(Array("ner_chunk"))
     .setOutputCol("sbert_embeddings")

val snomed_resolver = SentenceEntityResolverModel
     .pretrained("sbiobertresolve_snomed_findings_aux_concepts", "en", "clinical/models")
     .setInputCols(Array("ner_chunk", "sbert_embeddings"))
     .setOutputCol("snomed_code")

val snomed_pipelineModel= new PipelineModel().setStages(Array(document_assembler, sbert_embedder, snomed_resolver))

val snomed_lp = LightPipeline(snomed_pipelineModel)
val result = snomed_lp.fullAnnotate("atherosclerosis")
```
</div>

## Results

```bash
|    | chunks          | code     | resolutions                                                                                                                                                                                                                                                                                                                                                                                                                    | all_codes                                                                                                                                                                                          | all_k_aux_labels                                      | all_distances                                                                                                                                   |
|---:|:----------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | atherosclerosis | 38716007 | [atherosclerosis, atherosclerosis, atherosclerosis, atherosclerosis, atherosclerosis, atherosclerosis, atherosclerosis artery, coronary atherosclerosis, coronary atherosclerosis, coronary atherosclerosis, coronary atherosclerosis, coronary atherosclerosis, arteriosclerosis, carotid atherosclerosis, cardiovascular arteriosclerosis, aortic atherosclerosis, aortic atherosclerosis, atherosclerotic ischemic disease] | [38716007, 155382007, 155414001, 195251000, 266318005, 194848007, 441574008, 443502000, 41702007, 266231003, 155316000, 194841001, 28960008, 300920004, 39468009, 155415000, 195252007, 129573006] | 'Atherosclerosis', 'Observation', 'Morph Abnormality' | [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0280, 0.0451, 0.0451, 0.0451, 0.0451, 0.0451, 0.0462, 0.0477, 0.0466, 0.0490, 0.0490, 0.0485 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_snomed_findings_aux_concepts|
|Compatibility:|Healthcare NLP 3.1.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[snomed_code]|
|Language:|en|
|Case sensitive:|False|

## Data Source

http://www.snomed.org/
