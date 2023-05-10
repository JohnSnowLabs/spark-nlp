---
layout: model
title: Polish DistilBertForTokenClassification Cased model (from clarin-pl)
author: John Snow Labs
name: distilbert_token_classifier_fastpdn_distiluse
date: 2023-03-19
tags: [pl, open_source, distilbert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: pl
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `FastPDN-distiluse` is a Polish model originally trained by `clarin-pl`.

## Predicted Entities

`nam_fac_road`, `nam_pro_title_article`, `nam_fac_goe`, `nam_eve`, `nam_adj_country`, `nam_eve_human_holiday`, `nam_num_house`, `nam_org_company`, `nam_oth_currency`, `nam_fac_bridge`, `nam_liv_god`, `nam_fac_goe_stop`, `nam_pro_media_tv`, `nam_loc_gpe_admin3`, `nam_org_political_party`, `nam_oth`, `nam_pro_brand`, `nam_fac_park`, `nam_loc_gpe_city`, `nam_loc_hydronym_sea`, `nam_pro_media_web`, `nam_loc_gpe_conurbation`, `nam_loc_land_peak`, `nam_fac_system`, `nam_loc_gpe_district`, `nam_loc_land_island`, `nam_org_organization_sub`, `nam_loc_gpe_admin2`, `nam_adj_city`, `nam_liv_character`, `nam_pro_title_book`, `nam_loc_hydronym_lake`, `nam_loc_astronomical`, `nam_pro_award`, `nam_pro_title_tv`, `nam_loc`, `nam_loc_hydronym_river`, `nam_oth_position`, `nam_pro_vehicle`, `nam_org_institution`, `nam_pro_media`, `nam_pro_model_car`, `nam_org_group_team`, `nam_pro_software_game`, `nam_loc_land`, `nam_oth_tech`, `nam_loc_gpe_admin1`, `nam_adj_person`, `nam_loc_land_mountain`, `nam_liv_person`, `nam_eve_human_sport`, `nam_liv_animal`, `nam_oth_license`, `nam_oth_www`, `nam_loc_hydronym_ocean`, `nam_liv_habitant`, `nam_eve_human`, `nam_loc_land_continent`, `nam_org_nation`, `nam_pro_title_document`, `nam_pro_media_radio`, `nam_loc_country_region`, `nam_eve_human_cultural`, `nam_loc_hydronym`, `nam_loc_gpe_country`, `nam_oth_data_format`, `nam_num_phone`, `nam_loc_historical_region`, `nam_adj`, `nam_org_group_band`, `nam_pro_software`, `nam_pro_title_song`, `nam_loc_land_region`, `nam_pro`, `nam_org_organization`, `nam_pro_title_album`, `nam_org_group`, `nam_loc_gpe_subdivision`, `nam_pro_title_treaty`, `nam_fac_square`, `nam_pro_media_periodic`, `nam_pro_title`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_fastpdn_distiluse_pl_4.3.1_3.0_1679228450234.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_fastpdn_distiluse_pl_4.3.1_3.0_1679228450234.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_token_classifier_fastpdn_distiluse","pl") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols(Array("text")) 
    .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_token_classifier_fastpdn_distiluse","pl") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_token_classifier_fastpdn_distiluse|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|pl|
|Size:|509.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/clarin-pl/FastPDN-distiluse
- https://gitlab.clarin-pl.eu/information-extraction/poldeepner2
- https://gitlab.clarin-pl.eu/grupa-wieszcz/ner/fast-pdn
- https://clarin-pl.eu/dspace/bitstream/handle/11321/294/WytyczneKPWr-jednostkiidentyfikacyjne.pdf