---
layout: model
title: Company to Ticker using Nasdaq
author: John Snow Labs
name: finel_nasdaq_data_ticker
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

This is a Financial Entity Resolver model, trained to obtain the ticker from a Company Name, registered in NASDAQ. You can use this model after extracting a company name using any NER, and you will obtain its ticker.

After this, you can use `finmapper_nasdaq_data_ticker` to augment and obtain more information about a company using NASDAQ datasource, including Official Company Name, Sector, Location, Currency, etc.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/ER_EDGAR_CRUNCHBASE/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finel_nasdaq_data_ticker_en_1.0.0_3.0_1666473763228.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finel_nasdaq_data_ticker_en_1.0.0_3.0_1666473763228.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

use_er_model = finance.SentenceEntityResolverModel.pretrained('finel_nasdaq_data_ticker', 'en', 'finance/models')\
  .setInputCols("embeddings")\
  .setOutputCol('normalized')\

prediction_Model = PipelineModel(stages=[documentAssembler, use, use_er_model])

test_pred = prediction_Model.transform(testsdf).cache()
```

</div>

## Results

```bash
+----------------------+-------+
|text                  |result |
+----------------------+-------+
|FIDUS INVESTMENT corp |[FDUS] |
|ASPECT DEVELOPMENT Inc|[ASDV] |
|CFSB BANCORP          |[CFSB] |
|DALEEN TECHNOLOGIES   |[DALN1]|
|GLEASON Corporation   |[GLE1] |
+----------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finel_nasdaq_data_ticker|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings]|
|Output Labels:|[normalized]|
|Language:|en|
|Size:|69.8 MB|
|Case sensitive:|false|

## References

NASDAQ Database
