---
layout: model
title: Financial Relation Extraction (Tickers)
author: John Snow Labs
name: finre_has_ticker
date: 2022-10-15
tags: [en, finance, re, has_ticker, licensed]
task: Relation Extraction
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: RelationExtractionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model can be used to extract the Ticker of Companies or Product names. A Ticker (stock symbol) is a unique series of letters assigned to a security for trading purposes. For example: 

Company: Apple Inc.
Ticker: `AAPL`

## Predicted Entities

`has_ticker`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finre_has_ticker_en_1.0.0_3.0_1665842119957.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    .setWhiteList(['ORG'])

ner_model_ticker = finance.NerModel.pretrained("finner_ticker", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_ticker")

ner_converter_ticker = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "ner_ticker"]) \
    .setOutputCol("ner_chunk_ticker")

chunk_merger = finance.ChunkMergeApproach()\
    .setInputCols("ner_chunk_ticker", "ner_chunk_org")\
    .setOutputCol('ner_chunk')\
    .setMergeOverlapping(True)

pos = nlp.PerceptronModel.pretrained("pos_anc", 'en')\
     .setInputCols("sentence", "token")\
     .setOutputCol("pos")
    
dependency_parser = nlp.DependencyParserModel().pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos", "token"])\
    .setOutputCol("dependencies")

re_ner_chunk_filter = finance.RENerChunksFilter()\
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunk")\
    .setRelationPairs(["ORG-TICKER"])\
    .setMaxSyntacticDistance(4)

re_Model = finance.RelationExtractionDLModel.pretrained("finre_has_ticker", "en", "finance/models")\
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
    pos,
    dependency_parser,
    re_ner_chunk_filter,
    re_Model])

empty_df = spark.createDataFrame([['']]).toDF("text")
re_model = pipeline.fit(empty_df)
text="""'MTH - Meritage Homes Corporation Reports Disappointing Revenue. RECN, Resources Connection Inc. Shareholder Raymond James Trust Has Decreased Holding'"""
light_model = nlp.LightPipeline(re_model)
light_model.fullAnnotate(text)
```

</div>

## Results

```bash
|   relation | entity1 | entity1_begin | entity1_end | chunk1 | entity2 | entity2_begin | entity2_end |                     chunk2 | confidence |
|-----------:|--------:|--------------:|------------:|-------:|--------:|--------------:|------------:|---------------------------:|-----------:|
| has_ticker |  TICKER |             0 |           2 |    MTH |     ORG |             6 |          31 | Meritage Homes Corporation | 0.99532026 |
| has_ticker |  TICKER |            64 |          67 |   RECN |     ORG |            70 |          93 |   Resources Connection Inc | 0.97409964 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finre_has_ticker|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|409.9 MB|

## References

Manual annotations on tweets

## Benchmarking

```bash
label          Recall Precision  F1     Support     
has_ticker     0.717  0.827      0.768    60 
Avg.           0.717  0.827      0.768    -  
Weighted-Avg.  0.717  0.827      0.768    -  
```
