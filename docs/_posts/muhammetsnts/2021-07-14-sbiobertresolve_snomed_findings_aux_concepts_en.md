---
layout: model
title: Sentence Entity Resolver for Snomed Concepts
author: John Snow Labs
name: sbiobertresolve_snomed_findings_aux_concepts
date: 2021-07-14
tags: [snomed, licensed, en, clinical]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 3.1.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts to Snomed codes using sbiobert_base_cased_mli Sentence Bert Embeddings. This is also capable of extracting Morph Abnormality, Procedure, Substance, Physical Object, and Body Structure concepts of Snomed codes.

## Predicted Entities

Predicts Snomed Codes and their normalized definition for each chunk. In the metadata, the `all_k_aux_labels` can be divided to get further information: `ground truth`, `concept`, and `aux` .For example, in the example shared below the ground truth is `Malignant tumor of urinary bladder`, concept is `Condition`, and aux is `Clinical Finding`.

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

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli','en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sbert_embeddings")\
      .setCaseSensitive(True)

snomed_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_findings_aux_concepts", "en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("snomed_code")\
     .setDistanceFunction("EUCLIDEAN")

snomed_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        snomed_resolver])

snomed_lp = LightPipeline(snomed_pipelineModel)
result = snomed_lp.fullAnnotate("bladder cancer")
```
```scala
val document_assembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
     .setInputCols(["ner_chunk"])\
     .setOutputCol("sbert_embeddings")

val snomed_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_findings_aux_concepts", "en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("snomed_code")

val snomed_pipelineModel= new PipelineModel().setStages(Array(document_assembler, sbert_embedder, snomed_resolver))

val snomed_lp = LightPipeline(snomed_pipelineModel)
val result = snomed_lp.fullAnnotate("bladder cancer")
```
</div>

## Results

```bash
|    | chunks         | code      | resolutions                                                                                                                                                                                                                                                                                                                                                                                                                                    | all_codes                                                                                                                                                                       | billable_hcc_status_score                                       | all_distances                                                                                                                    |
|---:|:---------------|:----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
|  0 | bladder cancer | 399326009 | [bladder cancer, bladder cancer, invasive bladder cancer, carcinoma of bladder, carcinoma of bladder, carcinoma of bladder, superficial bladder cancer, adenocarcinoma of bladder, suspected bladder cancer, suspected bladder cancer, suspected bladder cancer, transitional cell carcinoma of bladder, transitional cell carcinoma of bladder, squamous cell carcinoma of bladder, cancer in situ of urinary bladder, tumor of bladder neck] | [399326009, 363455001, 425066001, 255108000, 269607003, 154540000, 425231005, 255110003, 139850000, 162582000, 315269000, 393562002, 255109008, 255111004, 92546004, 254932004] | ['Malignant tumor of urinary', 'Condition', 'Clinical Finding'] | [0.0000, 0.0000, 0.0539, 0.0666, 0.0666, 0.0666, 0.0809, 0.0881, 0.0904, 0.0904, 0.0904, 0.0880, 0.0880, 0.0913, 0.0978, 0.1080] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_snomed_findings_aux_concepts|
|Compatibility:|Spark NLP for Healthcare 3.1.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[snomed_code]|
|Language:|en|
|Case sensitive:|False|

## Data Source

http://www.snomed.org/
