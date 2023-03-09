---
layout: model
title: Mapping Companies to NASDAQ Stock Screener by Ticker
author: John Snow Labs
name: finmapper_nasdaq_ticker_stock_screener
date: 2023-01-19
tags: [en, finance, licensed, nasdaq, ticker]
task: Chunk Mapping
language: en
nav_key: models
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model allows you to, given a Ticker, get the following information about a company at Nasdaq Stock Screener:

 - Country
 - IPO_Year
 - Industry
 - Last_Sale
 - Market_Cap
 - Name
 - Net_Change
 - Percent_Change
 - Sector
 - Ticker
 - Volume

Firstly, you should get the TICKER symbol from the finance text with the `finner_ticker` model, then you can get detailed information about the company with the ChunkMapper model.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmapper_nasdaq_ticker_stock_screener_en_1.0.0_3.0_1674157233652.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finmapper_nasdaq_ticker_stock_screener_en_1.0.0_3.0_1674157233652.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol('text')\
    .setOutputCol('document')

tokenizer = nlp.Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained("finner_ticker", "en", "finance/models")\
    .setInputCols(["document", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["document", "token", "ner"])\
    .setOutputCol("ner_chunk")

CM = finance.ChunkMapperModel.pretrained('finmapper_nasdaq_ticker_stock_screener', 'en', 'finance/models')\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")

pipeline = nlp.Pipeline().setStages([document_assembler,
                                 tokenizer, 
                                 embeddings,
                                 ner_model, 
                                 ner_converter, 
                                 CM])
                                 
text = ["""There are some serious purchases and sales of AMZN stock today."""]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)

result = model.transform(test_data).select('mappings').collect()
```

</div>

## Results

```bash
"Country": "United States",
"IPO_Year": "1997",
"Industry": "Catalog/Specialty Distribution",
"Last_Sale": "$98.12",
"Market_Cap": "9.98556270184E11",
"Name": "Amazon.com Inc. Common Stock",
"Net_Change": "2.85",
"Percent_Change": "2.991%",
"Sector": "Consumer Discretionary",
"Ticker": "AMZN",
"Volume": "85412563"
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finmapper_nasdaq_ticker_stock_screener|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|584.5 KB|

## References

https://www.nasdaq.com/market-activity/stocks/screener
