---
layout: model
title: Financial Relation Extraction (Ticker Symbol)
author: John Snow Labs
name: finre_has_ticker
date: 2022-10-14
tags: [en, finance, re, ticker, licensed, open_source]
task: Relation Extraction
language: en
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model can be used to extract Stock Symbols(Tickers) of Companies. A stock symbol is a unique series of letters assigned to a security for trading purposes. Example:

Company : Apple Inc
Ticker : `AAPL`

## Predicted Entities

`has_ticker`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finre_has_ticker_en_4.2.0_3.0_1665751978333.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
       .setInputCol("text")\
       .setOutputCol("document")
        
sentence_detector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
      .setInputCols(["document"])\
      .setOutputCol("sentence")\
        
tokenizer = nlp.Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
      .setInputCols(["sentence", "token"]) \
      .setOutputCol("embeddings")

ner_model_org = finance.NerModel.pretrained("finner_orgs_prods_alias","en","finance/models")\
      .setInputCols(["sentence", "token", "embeddings"])\
      .setOutputCol("ner_org")

ner_converter_org = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner_org"])\
    .setOutputCol("ner_chunk_org")\
    .setWhiteList(['ORG'])\

ner_model_ticker = finance.NerModel.pretrained("finner_ticker", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_ticker")\

ner_converter_ticker = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "ner_ticker"]) \
    .setOutputCol("ner_chunk_ticker")\

chunk_merger = finance.ChunkMergeApproach()\
    .setInputCols("ner_chunk_ticker", "ner_chunk_org")\
    .setOutputCol('ner_chunk')\
    .setMergeOverlapping(True)\

re_Model = finance.RelationExtractionDLModel.load("REDL_hastick")\
     .setInputCols(["ner_chunk", "sentence"])\
     .setOutputCol("relations")\
     .setPredictionThreshold(0.2)

pipeline = Pipeline(stages=[
     document_assembler, 
     sentence_detector,
     tokenizer,
     embeddings,
     ner_model_org,
     ner_converter_org,
     ner_model_ticker,
     ner_converter_ticker,
     chunk_merger,
     re_Model])

empty_df = spark.createDataFrame([['']]).toDF("text")

re_model = pipeline.fit(empty_df)

text='''Report: NOV, National Oilwell Varco Inc. is planning a Whole Foods expansion to benefit Prime Now Warren Buffett'''

light_model = nlp.LightPipeline(re_model)

light_model.fullAnnotate(text)
```

</div>

## Results

```bash
|   relation | entity1 | entity1_begin | entity1_end | chunk1 | entity2 | entity2_begin | entity2_end |                      chunk2 | confidence |
|-----------|--------|--------------|------------|-------|--------|--------------|------------|----------------------------|-----------|
| has_ticker |  TICKER |             8 |          10 |    NOV |     ORG |            13 |          39 | National Oilwell Varco Inc. |  0.9966828 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finre_has_ticker|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.9 MB|

## References

Manual annotations on tweets

## Benchmarking

```bash
| Relation      | Recall Precision | F1    | Support |    |
|---------------|------------------|-------|---------|----|
| has_ticker    | 0.717            | 0.827 | 0.768   | 60 |
| Avg.          | 0.717            | 0.827 | 0.768   |    |
| Weighted Avg. | 0.717            | 0.827 | 0.768   |    |
```