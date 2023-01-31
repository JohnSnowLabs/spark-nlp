---
layout: model
title: English DistilBertForSequenceClassification Cased model (from sampathkethineedi)
author: John Snow Labs
name: distilbert_sequence_classifier_industry_classification
date: 2022-08-23
tags: [distilbert, sequence_classification, open_source, en]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `industry-classification` is a English model originally trained by `sampathkethineedi`.

## Predicted Entities

`Semiconductors`, `Data Processing & Outsourced Services`, `Oil & Gas Exploration & Production`, `Industrial Machinery`, `Technology Distributors`, `Apparel Retail`, `Application Software`, `Research & Consulting Services`, `Specialty Stores`, `Diversified Support Services`, `Gold`, `Human Resource & Employment Services`, `Interactive Media & Services`, `Internet & Direct Marketing Retail`, `Auto Parts & Equipment`, `Building Products`, `Personal Products`, `Communications Equipment`, `Electronic Equipment & Instruments`, `Regional Banks`, `Systems Software`, `Health Care Services`, `Health Care Supplies`, `Asset Management & Custody Banks`, `Aerospace & Defense`, `Specialty Chemicals`, `Life Sciences Tools & Services`, `Electric Utilities`, `Commodity Chemicals`, `Health Care Equipment`, `Technology Hardware, Storage & Peripherals`, `Construction Machinery & Heavy Trucks`, `Environmental & Facilities Services`, `Oil & Gas Equipment & Services`, `Oil & Gas Refining & Marketing`, `Casinos & Gaming`, `Diversified Metals & Mining`, `Property & Casualty Insurance`, `Hotels, Resorts & Cruise Lines`, `IT Consulting & Other Services`, `Leisure Products`, `Pharmaceuticals`, `Movies & Entertainment`, `Restaurants`, `Steel`, `Thrifts & Mortgage Finance`, `Health Care Facilities`, `Oil & Gas Storage & Transportation`, `Internet Services & Infrastructure`, `Health Care Technology`, `Packaged Foods & Meats`, `Integrated Telecommunication Services`, `Consumer Finance`, `Investment Banking & Brokerage`, `Electrical Components & Equipment`, `Trading Companies & Distributors`, `Construction & Engineering`, `Advertising`, `Homebuilding`, `Biotechnology`, `Real Estate Operating Companies`, `Apparel, Accessories & Luxury Goods`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_industry_classification_en_4.1.0_3.0_1661277890942.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_industry_classification_en_4.1.0_3.0_1661277890942.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_industry_classification","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_industry_classification","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_sequence_classifier_industry_classification|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|249.9 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/sampathkethineedi/industry-classification