---
layout: model
title: Resolver Company Names to Tickers using Nasdaq Stock Screener
author: John Snow Labs
name: finel_nasdaq_name2ticker_stock_screener
date: 2023-01-20
tags: [en, finance, licensed, nasdaq, ticker, company]
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

This is an Entity Resolution / Entity Linking model, which is able to provide Ticker / Trading Symbols using a Company Name as an input. You can use any NER which extracts Organizations / Companies / Parties to then send the input to `finel_nasdaq_company_name_stock_screener` model to get normalized company name. Finally, this Entity Linking model get the Ticker / Trading Symbol (given the company has one).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finel_nasdaq_name2ticker_stock_screener_en_1.0.0_3.0_1674228715730.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finel_nasdaq_name2ticker_stock_screener_en_1.0.0_3.0_1674228715730.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("sentence")

use = nlp.UniversalSentenceEncoder.pretrained("tfhub_use", "en")\
    .setInputCols(["sentence"])\
    .setOutputCol("embeddings")

use_er_model = finance.SentenceEntityResolverModel.pretrained('finel_nasdaq_name2ticker_stock_screener', 'en', 'finance/models')\
  .setInputCols("embeddings")\
  .setOutputCol('normalized')\

pipeline = nlp.PipelineModel(stages=[documentAssembler, use, use_er_model])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

lp = nlp.LightPipeline(model)

result = lp.annotate("Nike Inc. Common Stock")
```

</div>

## Results

```bash
['NKE']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finel_nasdaq_name2ticker_stock_screener|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[normalized]|
|Language:|en|
|Size:|54.6 MB|
|Case sensitive:|false|

## References

https://www.nasdaq.com/market-activity/stocks/screener