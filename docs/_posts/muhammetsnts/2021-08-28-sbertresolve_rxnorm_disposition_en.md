---
layout: model
title: Sentence Entity Resolver for RxNorm (sbertresolve_rxnorm_disposition)
author: John Snow Labs
name: sbertresolve_rxnorm_disposition
date: 2021-08-28
tags: [rxnorm, en, licensed, clinical]
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

This model maps medication entities (like drugs/ingredients) to RxNorm codes and their dispositions using sbert_jsl_medium_uncased Sentence Bert Embeddings. If you look for a faster inference with just drug names (excluding dosage and strength), this version of RxNorm model would be a better alternative.

## Predicted Entities

Predicts RxNorm Codes, their normalized definition for each chunk, and dispositions if any. In the result, look for the aux_label parameter in the metadata to get dispositions divided by `|`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbertresolve_rxnorm_disposition_en_3.1.3_2.4_1630175088499.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_rxnorm_disposition", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("rxnorm_code")\
.setDistanceFunction("EUCLIDEAN")

rxnorm_pipelineModel = PipelineModel(
stages = [
documentAssembler,
sbert_embedder,
rxnorm_resolver])

rxnorm_lp = LightPipeline(rxnorm_pipelineModel)
result = rxnorm_lp.fullAnnotate("alizapride 25 mg/ml")
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbert_jsl_medium_uncased", "en","clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sbert_embeddings")

val rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_rxnorm_disposition", "en", "clinical/models") 
.setInputCols(Array("ner_chunk", "sbert_embeddings")) 
.setOutputCol("rxnorm_code")
.setDistanceFunction("EUCLIDEAN")

val pipelineModel= new PipelineModel().setStages(Array(documentAssembler, sbert_embedder, rxnorm_resolver))

val rxnorm_lp = LightPipeline(pipelineModel)
val result = rxnorm_lp.fullAnnotate("alizapride 25 mg/ml")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.rxnorm.disposition").predict("""alizapride 25 mg/ml""")
```

</div>

## Results

```bash
|    | chunks             | code   | resolutions                                                                                                                                                                            | all_codes                                                       | all_k_aux_labels                                                                                            | all_distances                                                 |
|---:|:-------------------|:-------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
|  0 |alizapride 25 mg/ml | 330948 | [alizapride 25 mg/ml, alizapride 50 mg, alizapride 25 mg/ml oral solution, adalimumab 50 mg/ml, adalimumab 100 mg/ml [humira], adalimumab 50 mg/ml [humira], alirocumab 150 mg/ml, ...]| [330948, 330949, 249531, 358817, 1726845, 576023, 1659153, ...] | [Dopamine receptor antagonist, Dopamine receptor antagonist, Dopamine receptor antagonist, -, -, -, -, ...] | [0.0000, 0.0936, 0.1166, 0.1525, 0.1584, 0.1567, 0.1631, ...] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbertresolve_rxnorm_disposition|
|Compatibility:|Healthcare NLP 3.1.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk, sbert_embeddings]|
|Output Labels:|[new_code]|
|Language:|en|
|Case sensitive:|false|