---
layout: model
title: DistilBERT Sequence Classification - Industry (distilbert_sequence_classifier_industry)
author: John Snow Labs
name: distilbert_sequence_classifier_industry
date: 2021-11-21
tags: [sequence_classification, distilbert, en, english, open_source, industry, business]
task: Text Classification
language: en
edition: Spark NLP 3.3.3
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`distilbert_sequence_classifier_industry` Model to classify a business description into one of 62 industry tags. Trained on 7000 samples of Business Descriptions and associated labels of companies in India.

## Predicted Entities

`Advertising`, `Aerospace & Defense`, `Apparel Retail`, `Apparel`, `Accessories & Luxury Goods`, `Application Software`, `Asset Management & Custody Banks`, `Auto Parts & Equipment`, `Biotechnology`, `Building Products`, `Casinos & Gaming`, `Commodity Chemicals`, `Communications Equipment`, `Construction & Engineering`, `Construction Machinery & Heavy Trucks`, `Consumer Finance`, `Data Processing & Outsourced Services`, `Diversified Metals & Mining`, `Diversified Support Services`, `Electric Utilities`, `Electrical Components & Equipment`, `Electronic Equipment & Instruments`, `Environmental & Facilities Services`, `Gold`, `Health Care Equipment`, `Health Care Facilities`, `Health Care Services`, `Health Care Supplies`, `Health Care Technology`, `Homebuilding`, `Hotels`, `Resorts & Cruise Lines`, `Human Resource & Employment Services`, `IT Consulting & Other Services`, `Industrial Machinery`, `Integrated Telecommunication Services`, `Interactive Media & Services`, `Internet & Direct Marketing Retail`, `Internet Services & Infrastructure`, `Investment Banking & Brokerage`, `Leisure Products`, `Life Sciences Tools & Services`, `Movies & Entertainment`, `Oil & Gas Equipment & Services`, `Oil & Gas Exploration & Production`, `Oil & Gas Refining & Marketing`, `Oil & Gas Storage & Transportation`, `Packaged Foods & Meats`, `Personal Products`, `Pharmaceuticals`, `Property & Casualty Insurance`, `Real Estate Operating Companies`, `Regional Banks`, `Research & Consulting Services`, `Restaurants`, `Semiconductors`, `Specialty Chemicals`, `Specialty Stores`, `Steel`, `Systems Software`, `Technology Distributors`, `Technology Hardware`, `Storage & Peripherals`, `Thrifts & Mortgage Finance`, `Trading Companies & Distributors`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_industry_en_3.3.3_3.0_1637496932885.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol('text') \
.setOutputCol('document')

tokenizer = Tokenizer() \
.setInputCols(['document']) \
.setOutputCol('token')

sequenceClassifier = DistilBertForSequenceClassification \
.pretrained('distilbert_sequence_classifier_industry', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier    
])

example = spark.createDataFrame([['Stellar Capital Services Limited is an India-based non-banking financial company ... loan against property, management consultancy, personal loans and unsecured loans.']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_industry", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("Stellar Capital Services Limited is an India-based non-banking financial company ... loan against property, management consultancy, personal loans and unsecured loans.").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.distilbert_sequence.industry").predict("""Stellar Capital Services Limited is an India-based non-banking financial company ... loan against property, management consultancy, personal loans and unsecured loans.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_sequence_classifier_industry|
|Compatibility:|Spark NLP 3.3.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/sampathkethineedi/industry-classification](https://huggingface.co/sampathkethineedi/industry-classification)
