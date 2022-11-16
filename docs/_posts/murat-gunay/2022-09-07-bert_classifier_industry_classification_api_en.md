---
layout: model
title: English BertForSequenceClassification Cased model (from sampathkethineedi)
author: John Snow Labs
name: bert_classifier_industry_classification_api
date: 2022-09-07
tags: [en, open_source, bert, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `industry-classification-api` is a English model originally trained by `sampathkethineedi`.

## Predicted Entities

`Casinos & Gaming`, `Apparel, Accessories & Luxury Goods`, `Research & Consulting Services`, `Restaurants`, `Oil & Gas Equipment & Services`, `Consumer Finance`, `Industrial Machinery`, `Health Care Technology`, `Specialty Chemicals`, `Regional Banks`, `Auto Parts & Equipment`, `Biotechnology`, `Construction Machinery & Heavy Trucks`, `Interactive Media & Services`, `Internet Services & Infrastructure`, `Systems Software`, `Gold`, `Packaged Foods & Meats`, `Construction & Engineering`, `Asset Management & Custody Banks`, `Data Processing & Outsourced Services`, `Pharmaceuticals`, `Specialty Stores`, `Oil & Gas Storage & Transportation`, `Technology Hardware, Storage & Peripherals`, `Movies & Entertainment`, `Personal Products`, `Oil & Gas Exploration & Production`, `Health Care Facilities`, `Commodity Chemicals`, `Real Estate Operating Companies`, `Human Resource & Employment Services`, `Health Care Equipment`, `Communications Equipment`, `Oil & Gas Refining & Marketing`, `Aerospace & Defense`, `Leisure Products`, `Apparel Retail`, `Diversified Support Services`, `Electric Utilities`, `Hotels, Resorts & Cruise Lines`, `Life Sciences Tools & Services`, `Diversified Metals & Mining`, `Building Products`, `Investment Banking & Brokerage`, `Semiconductors`, `Application Software`, `Internet & Direct Marketing Retail`, `Health Care Services`, `Homebuilding`, `Trading Companies & Distributors`, `Advertising`, `Environmental & Facilities Services`, `Steel`, `Integrated Telecommunication Services`, `Health Care Supplies`, `Electrical Components & Equipment`, `Thrifts & Mortgage Finance`, `Technology Distributors`, `Electronic Equipment & Instruments`, `Property & Casualty Insurance`, `IT Consulting & Other Services`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_industry_classification_api_en_4.1.0_3.0_1662513488095.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_industry_classification_api","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_industry_classification_api","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_industry_classification_api|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.2 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/sampathkethineedi/industry-classification-api