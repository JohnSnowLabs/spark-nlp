---
layout: model
title: Sentence Entity Resolver for Snomed (sbertresolve_snomed_conditions)
author: John Snow Labs
name: sbertresolve_snomed_conditions
date: 2021-08-28
tags: [snomed, licensed, en, clinical]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.1.3
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities (domain: Conditions) to Snomed codes using `sbert_jsl_medium_uncased` Sentence Bert Embeddings.

## Predicted Entities

Snomed Codes and their normalized definition with `sbert_jsl_medium_uncased ` embeddings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbertresolve_snomed_conditions_en_3.1.3_2.4_1630180858399.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbertresolve_snomed_conditions_en_3.1.3_2.4_1630180858399.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbert_jsl_medium_uncased', 'en','clinical/models')\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

snomed_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_snomed_conditions", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("snomed_code")\
.setDistanceFunction("EUCLIDEAN")

snomed_pipelineModel = PipelineModel(
stages = [
documentAssembler,
sbert_embedder,
snomed_resolver
])

snomed_lp = LightPipeline(snomed_pipelineModel)
result = snomed_lp.fullAnnotate("schizophrenia")
```
```scala
val documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained('sbert_jsl_medium_uncased', 'en','clinical/models')\
.setInputCols("ner_chunk")\
.setOutputCol("sbert_embeddings")

val snomed_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_snomed_conditions", "en", "clinical/models") \
.setInputCols(Array("ner_chunk", "sbert_embeddings")) \
.setOutputCol("snomed_code")\
.setDistanceFunction("EUCLIDEAN")

val snomed_pipelineModel = new PipelineModel().setStages(Array(documentAssembler,sbert_embedder,snomed_resolver))

val snomed_lp = LightPipeline(snomed_pipelineModel)
val result = snomed_lp.fullAnnotate("schizophrenia")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.snomed_conditions").predict("""Put your text here.""")
```

</div>

## Results

```bash
|    | chunks        | code     | resolutions                                                                                                              | all_codes                                                            | all_distances                                        |
|---:|:--------------|:---------|:-------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------|:-----------------------------------------------------|
|  0 | schizophrenia | 58214004 | [schizophrenia, chronic schizophrenia, borderline schizophrenia, schizophrenia, catatonic, subchronic schizophrenia, ...]| [58214004, 83746006, 274952002, 191542003, 191529003, 16990005, ...] | 0.0000, 0.0774, 0.0838, 0.0927, 0.0970, 0.0970, ...] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbertresolve_snomed_conditions|
|Compatibility:|Healthcare NLP 3.1.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk, sbert_embeddings]|
|Output Labels:|[snomed_code]|
|Language:|en|
|Case sensitive:|false|