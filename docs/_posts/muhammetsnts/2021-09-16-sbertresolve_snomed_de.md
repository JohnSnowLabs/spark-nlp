---
layout: model
title: Sentence Entity Resolver for Snomed Codes
author: John Snow Labs
name: sbertresolve_snomed
date: 2021-09-16
tags: [snomed, de, clinial, licensed]
task: Entity Resolution
language: de
edition: Healthcare NLP 3.2.2
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical entities to SNOMED codes for the German language using `sent_bert_base_cased` (de) embeddings.

## Predicted Entities

`SNOMED Codes`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_SNOMED_DE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/14.German_Healthcare_Models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbertresolve_snomed_de_3.2.2_2.4_1631826969583.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbertresolve_snomed_de_3.2.2_2.4_1631826969583.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "de")\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

snomed_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_snomed", "de", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("snomed_code")

snomed_pipelineModel = PipelineModel(
stages = [
documentAssembler,
sbert_embedder,
snomed_resolver])

snomed_lp = LightPipeline(snomed_pipelineModel)

```
```scala
val documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "de")\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

val snomed_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_snomed", "de", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("snomed_code")

val snomed_pipelineModel = new PipelineModel().setStages(Array(documentAssembler, sbert_embedder, snomed_resolver))

val snomed_lp = LightPipeline(snomed_pipelineModel)
```


{:.nlu-block}
```python
import nlu
nlu.load("de.resolve.snomed").predict("""Put your text here.""")
```

</div>

## Results

```bash
|    | chunks            | code    | resolutions                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | all_codes                                                                                                                                                                          | all_distances                                                                                                                                                                                            |
|---:|:------------------|:--------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | Bronchialkarzinom | 22628   | Bronchialkarzinom, Bronchuskarzinom, Rektumkarzinom, Klavikulakarzinom, Lippenkarzinom, Urothelkarzinom, Hodenteratokarzinom, Unterbauchkarzinom, Teratokarzinom, Oropharynxkarzinom, Harnleiterkarzinom, Herzbeutelkarzinom, Thekazellkarzinom, Plattenepithelkarzinom, Weichteilkarzinom, Perikardkarzinom, Zervixkarzinom, Samenstrangkarzinom, Nierenkelchkarzinom, Querkolonkarzinom, Perianalkarzinom, Endozervixkarzinom, Parotiskarzinom, Gehörgangskarzinom, Prostatakarzinom| [22628, 111139, 18116, 107569, 18830, 22909, 16259, 111193, 22383, 19807, 22613, 20014, 74820, 21331, 30182, 20015, 23130, 22068, 20340, 29968, 15757, 23917, 25303, 17800, 21706] | [0.0000, 0.0073, 0.0090, 0.0098, 0.0098, 0.0102, 0.0102, 0.0110, 0.0111, 0.0120, 0.0121, 0.0123, 0.0128, 0.0130, 0.0129, 0.0131, 0.0128, 0.0131, 0.0135, 0.0133, 0.0137, 0.0137, 0.0139, 0.0137, 0.0139] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbertresolve_snomed|
|Compatibility:|Healthcare NLP 3.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[snomed_code]|
|Language:|de|
|Case sensitive:|false|
