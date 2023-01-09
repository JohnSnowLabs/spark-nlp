---
layout: model
title: Company Name Normalization using Nasdaq
author: John Snow Labs
name: finel_nasdaq_data_company_name
date: 2022-10-22
tags: [en, finance, companies, nasdaq, ticker, licensed]
task: Entity Resolution
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Financial Entity Resolver model, trained to obtain normalized versions of Company Names, registered in NASDAQ. You can use this model after extracting a company name using any NER, and you will obtain the official name of the company as per NASDAQ database.

After this, you can use `finmapper_nasdaq_data_company_name` to augment and obtain more information about a company using NASDAQ datasource, including Ticker, Sector, Location, Currency, etc.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/ER_EDGAR_CRUNCHBASE/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finel_nasdaq_data_company_name_en_1.0.0_3.0_1666473632696.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
test = ["FIDUS INVESTMENT corp","ASPECT DEVELOPMENT Inc","CFSB BANCORP","DALEEN TECHNOLOGIES","GLEASON Corporation"]
testdf = pandas.DataFrame(test, columns=['text'])
testsdf = spark.createDataFrame(testdf).toDF('text')

documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("sentence")

use = nlp.UniversalSentenceEncoder.pretrained("tfhub_use_lg", "en")\
    .setInputCols(["sentence"])\
    .setOutputCol("embeddings")

use_er_model = finance.SentenceEntityResolverModel.pretrained('finel_nasdaq_data_company_name', 'en', 'finance/models')\
  .setInputCols("embeddings")\
  .setOutputCol('normalized')\

prediction_Model = PipelineModel(stages=[documentAssembler, use, use_er_model])

test_pred = prediction_Model.transform(testsdf)
```

</div>

## Results

```bash
+----------------------+-------------------------+
|text                  |result                   |
+----------------------+-------------------------+
|FIDUS INVESTMENT corp |[FIDUS INVESTMENT CORP]  |
|ASPECT DEVELOPMENT Inc|[ASPECT DEVELOPMENT INC] |
|CFSB BANCORP          |[CFSB BANCORP INC]       |
|DALEEN TECHNOLOGIES   |[DALEEN TECHNOLOGIES INC]|
|GLEASON Corporation   |[GLEASON CORP]           |
+----------------------+-------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finel_nasdaq_data_company_name|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings]|
|Output Labels:|[normalized]|
|Language:|en|
|Size:|69.7 MB|
|Case sensitive:|false|

## References

NASDAQ Database
