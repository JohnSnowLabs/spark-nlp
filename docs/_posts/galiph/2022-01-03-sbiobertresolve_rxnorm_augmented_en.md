---
layout: model
title: Sentence Entity Resolver for RxNorm (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_rxnorm_augmented
date: 2022-01-03
tags: [rxnorm, licensed, en, clinical, entity_resolution]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.1
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It trained on the augmented version of the dataset which is used in previous RxNorm resolver models. Additionally, this model returns concept classes of the drugs in `all_k_aux_labels` column.

## Predicted Entities

`RxNorm Codes`, `Concept Classes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_augmented_en_3.3.1_2.4_1641241820334.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_augmented_en_3.3.1_2.4_1641241820334.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

```sbiobertresolve_rxnorm_augmented``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_posology``` as NER model. ```DRUG``` set in ```.setWhiteList()```.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

rxnorm_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_rxnorm_augmented", "en", "clinical/models")\
.setInputCols(["ner_chunk", "sbert_embeddings"])\
.setOutputCol("rxnorm_code")\
.setDistanceFunction("EUCLIDEAN")

rxnorm_pipeline = Pipeline(
stages = [
documentAssembler,
sbert_embedder,
rxnorm_resolver])

light_model = LightPipeline(rxnorm_pipeline)

result = light_model.fullAnnotate(["Coumadin 5 mg", "aspirin", ""avandia 4 mg"])
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sbert_embeddings")

val rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented_cased", "en", "clinical/models") \
.setInputCols(Array("ner_chunk", "sbert_embeddings")) \
.setOutputCol("rxnorm_code")\
.setDistanceFunction("EUCLIDEAN")

val rxnorm_pipelineModel = new PipelineModel().setStages(Array(documentAssembler, sbert_embedder, rxnorm_resolver))

val light_model = LightPipeline(rxnorm_pipelineModel)

val result = light_model.fullAnnotate(Array("Coumadin 5 mg", "aspirin", ""avandia 4 mg"))
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.rxnen.med_ner.deid_subentityorm_augmented").predict("""Coumadin 5 mg""")
```

</div>

## Results

```bash
|    |   RxNormCode | Resolution                               | all_k_results                     | all_k_distances                   | all_k_cosine_distances            | all_k_resolutions                                               | all_k_aux_labels                  |
|---:|-------------:|:-----------------------------------------|:----------------------------------|:----------------------------------|:----------------------------------|:----------------------------------------------------------------|:----------------------------------|
|  0 |       855333 | warfarin sodium 5 MG [Coumadin]          | 855333:::432467:::438740:::103... | 3.0367:::4.7790:::4.7790:::5.3... | 0.0161:::0.0395:::0.0395:::0.0... | warfarin sodium 5 MG [Coumadin]:::coumarin 5 MG Oral Tablet:... | Branded Drug Comp:::Clinical D... |
|  1 |      1537020 | aspirin Effervescent Oral Tablet         | 1537020:::1191:::1295740:::405... | 0.0000:::0.0000:::4.1826:::5.7... | 0.0000:::0.0000:::0.0292:::0.0... | aspirin Effervescent Oral Tablet:::aspirin:::aspirin Oral Po... | Clinical Drug Form:::Ingredien... |
|  2 |       261242 | rosiglitazone 4 MG Oral Tablet [Avandia] | 261242:::810073:::153845:::109... | 0.0000:::4.7482:::5.0125:::5.2... | 0.0000:::0.0365:::0.0409:::0.0... | rosiglitazone 4 MG Oral Tablet [Avandia]:::fesoterodine fuma... | Branded Drug:::Branded Drug Co... |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_rxnorm_augmented|
|Compatibility:|Healthcare NLP 3.3.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Size:|976.1 MB|
|Case sensitive:|false|