---
layout: model
title: English BertForTokenClassification Cased model (from ramybaly)
author: John Snow Labs
name: bert_ner_ner_nerd_fine
date: 2022-08-03
tags: [bert, ner, open_source, en]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `ner_nerd_fine` is a English model originally trained by `ramybaly`.

## Predicted Entities

`MISC_educationaldegree`, `ORG_other`, `BUILDING_restaurant`, `MISC_law`, `LOC_mountain`, `ART_other`, `MISC_medical`, `LOC_other`, `PER_athlete`, `PRODUCT_food`, `MISC_god`, `BUILDING_theater`, `LOC_GPE`, `ORG_media/newspaper`, `PRODUCT_other`, `ORG_government/governmentagency`, `PRODUCT_airplane`, `PRODUCT_software`, `BUILDING_other`, `ART_film`, `LOC_park`, `LOC_road/railway/highway/transit`, `PER_soldier`, `PRODUCT_weapon`, `EVENT_other`, `ORG_sportsleague`, `PRODUCT_train`, `PER_other`, `PER_politician`, `EVENT_election`, `ORG_company`, `PER_director`, `BUILDING_sportsfacility`, `ART_painting`, `BUILDING_airport`, `ART_music`, `LOC_island`, `ORG_politicalparty`, `MISC_award`, `PRODUCT_ship`, `BUILDING_hospital`, `ORG_sportsteam`, `MISC_livingthing`, `MISC_astronomything`, `BUILDING_hotel`, `MISC_language`, `EVENT_attack/battle/war/militaryconflict`, `LOC_bodiesofwater`, `EVENT_sportsevent`, `ORG_religion`, `PRODUCT_car`, `BUILDING_library`, `ORG_education`, `MISC_disease`, `MISC_currency`, `PER_scholar`, `EVENT_disaster`, `PRODUCT_game`, `PER_artist/author`, `ART_writtenart`, `EVENT_protest`, `MISC_chemicalthing`, `PER_actor`, `MISC_biologything`, `ART_broadcastprogram`, `ORG_showorganization`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_ner_nerd_fine_en_4.1.0_3.0_1659515006113.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_ner_nerd_fine","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_ner_nerd_fine","en") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_ner_nerd_fine|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|407.9 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/ramybaly/ner_nerd_fine