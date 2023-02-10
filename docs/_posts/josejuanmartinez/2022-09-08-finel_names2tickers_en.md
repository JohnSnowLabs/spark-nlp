---
layout: model
title: Resolver Company Names to Tickers
author: John Snow Labs
name: finel_names2tickers
date: 2022-09-08
tags: [en, finance, companies, tickers, nasdaq, licensed]
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

This is an Entity Resolution / Entity Linking model, which is able to provide Ticker / Trading Symbols using a Company Name as an input. You can use any NER which extracts Organizations / Companies / Parties to then send the output to this Entity Linking model and get the Ticker / Trading Symbol (given the company has one).

## Predicted Entities



{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/financial_company_normalization){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finel_names2tickers_en_1.0.0_3.2_1662636940877.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finel_names2tickers_en_1.0.0_3.2_1662636940877.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

embeddings = nlp.UniversalSentenceEncoder.pretrained("tfhub_use", "en") \
      .setInputCols("ner_chunk") \
      .setOutputCol("sentence_embeddings")
    
resolver = finance.SentenceEntityResolverModel.pretrained("finel_names2tickers", "en", "finance/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("name")\
      .setDistanceFunction("EUCLIDEAN")

pipelineModel = PipelineModel(
      stages = [
          documentAssembler,
          embeddings,
          resolver])

lp = LightPipeline(pipelineModel)

lp.fullAnnotate("apple")
```

</div>

## Results

```bash
+--------+---------+----------------------------------------------+-------------------------+-----------------------------------+
|   chunk|    code |                                     all_codes|             resolutions |                      all_distances|
+--------+---------+----------------------------------------------+-------------------------+-----------------------------------+
|  apple |   Apple | [Apple, Apple inc., Apple INC. , Apple Inc.] |[AAPL, AAPL, AAPL, AAPL] |  [0.0000, 0.1093, 0.1093, 0.1093] |
+--------+---------+----------------------------------------------+-------------------------+-----------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finel_names2tickers|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[company_ticker]|
|Language:|en|
|Size:|20.3 MB|
|Case sensitive:|false|

## References

https://data.world/johnsnowlabs/list-of-companies-in-nasdaq-exchanges
