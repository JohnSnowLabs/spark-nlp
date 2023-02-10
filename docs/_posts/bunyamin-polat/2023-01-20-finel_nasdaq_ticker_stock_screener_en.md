---
layout: model
title: Resolver Company Names to Tickers using Nasdaq Stock Screener
author: John Snow Labs
name: finel_nasdaq_ticker_stock_screener
date: 2023-01-20
tags: [en, licensed, finance, nasdaq, ticker]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finel_nasdaq_ticker_stock_screener_en_1.0.0_3.0_1674236954508.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finel_nasdaq_ticker_stock_screener_en_1.0.0_3.0_1674236954508.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en")\
    .setInputCols(["document", "token"])\
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained('finner_orgs_prods_alias', 'en', 'finance/models')\
    .setInputCols(["document", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["document", "token", "ner"])\
    .setOutputCol("ner_chunk")

chunkToDoc = nlp.Chunk2Doc()\
    .setInputCols("ner_chunk")\
    .setOutputCol("ner_chunk_doc") 

ticker_embeddings = nlp.UniversalSentenceEncoder.pretrained("tfhub_use", "en")\
    .setInputCols("ner_chunk_doc")\
    .setOutputCol("ticker_embeddings")

er_ticker_model = finance.SentenceEntityResolverModel.pretrained('finel_nasdaq_ticker_stock_screener', 'en', 'finance/model')\
    .setInputCols(["ticker_embeddings"])\
    .setOutputCol("ticker")\
    .setAuxLabelCol("company_name")

pipeline = nlp.Pipeline().setStages([document_assembler,
                              tokenizer, 
                              ner_embeddings,
                              ner_model, 
                              ner_converter,
                              chunkToDoc,
                              ticker_embeddings,
                              er_ticker_model])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

lp = nlp.LightPipeline(model)

text = """Nike is an American multinational association that is involved in the design, development, manufacturing and worldwide marketing and sales of apparel, footwear, accessories, equipment and services."""

result = lp.annotate(text)

result["ticker"]
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
|Model Name:|finel_nasdaq_ticker_stock_screener|
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
