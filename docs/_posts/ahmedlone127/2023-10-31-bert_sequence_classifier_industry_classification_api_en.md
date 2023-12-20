---
layout: model
title: English BertForSequenceClassification Cased model (from sampathkethineedi)
author: John Snow Labs
name: bert_sequence_classifier_industry_classification_api
date: 2023-10-31
tags: [en, open_source, bert, sequence_classification, ner, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `industry-classification-api` is a English model originally trained by `sampathkethineedi`.

## Predicted Entities

`Electronic Equipment & Instruments`, `Commodity Chemicals`, `Leisure Products`, `Health Care Services`, `Regional Banks`, `Life Sciences Tools & Services`, `Industrial Machinery`, `Gold`, `Investment Banking & Brokerage`, `Restaurants`, `Packaged Foods & Meats`, `IT Consulting & Other Services`, `Oil & Gas Equipment & Services`, `Health Care Supplies`, `Aerospace & Defense`, `Human Resource & Employment Services`, `Application Software`, `Property & Casualty Insurance`, `Movies & Entertainment`, `Oil & Gas Storage & Transportation`, `Apparel Retail`, `Electrical Components & Equipment`, `Consumer Finance`, `Construction Machinery & Heavy Trucks`, `Advertising`, `Casinos & Gaming`, `Construction & Engineering`, `Systems Software`, `Auto Parts & Equipment`, `Data Processing & Outsourced Services`, `Specialty Stores`, `Research & Consulting Services`, `Oil & Gas Exploration & Production`, `Pharmaceuticals`, `Interactive Media & Services`, `Homebuilding`, `Building Products`, `Personal Products`, `Electric Utilities`, `Communications Equipment`, `Trading Companies & Distributors`, `Health Care Equipment`, `Semiconductors`, `Internet & Direct Marketing Retail`, `Environmental & Facilities Services`, `Thrifts & Mortgage Finance`, `Diversified Metals & Mining`, `Oil & Gas Refining & Marketing`, `Steel`, `Diversified Support Services`, `Technology Distributors`, `Health Care Facilities`, `Health Care Technology`, `Biotechnology`, `Integrated Telecommunication Services`, `Real Estate Operating Companies`, `Internet Services & Infrastructure`, `Asset Management & Custody Banks`, `Specialty Chemicals`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_industry_classification_api_en_5.1.4_3.4_1698795567583.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_industry_classification_api_en_5.1.4_3.4_1698795567583.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_industry_classification_api","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_industry_classification_api","en")
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_industry_classification_api|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|409.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

References

- https://huggingface.co/sampathkethineedi/industry-classification-api