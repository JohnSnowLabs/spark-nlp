---
layout: model
title: Resolve Tickers to Company Names
author: John Snow Labs
name: finel_tickers2names
date: 2022-09-09
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

This is an Entity Resolution / Entity Linking model, which is able to provide Company Names given their Ticker / Trading Symbols. You can use any NER which extracts Tickersto then send the output to this Entity Linking model and get the Company Name.

## Predicted Entities



{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/financial_company_normalization){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finel_tickers2names_en_1.0.0_3.2_1662733866127.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    
resolver = finance.SentenceEntityResolverModel.pretrained("finel_tickers2names", "en", "finance/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("name")\
      .setDistanceFunction("EUCLIDEAN")

pipelineModel = PipelineModel(
      stages = [
          documentAssembler,
          embeddings,
          resolver])

lp = LightPipeline(pipelineModel)

lp.fullAnnotate("unit")

```

</div>

## Results

```bash
+-------+--------------------+-----------------------------------------------------------------+----------------------------------------------------------------+---------------------------+
|  chunk|               code |                                                        all_codes|                                                    resolutions |              all_distances|
+-------+--------------------+-----------------------------------------------------------------+----------------------------------------------------------------+---------------------------+
|  unit |   UNITI GROUP INC. | [UNITI GROUP INC., Uniti Group INC. , Uniti Group Incorporated] |[UNITI GROUP INC., Uniti Group INC. , Uniti Group Incorporated] |  [0.0000, 0.0000, 0.0000] |
+-------+--------------------+-----------------------------------------------------------------+----------------------------------------------------------------+---------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finel_tickers2names|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[org_company_name]|
|Language:|en|
|Size:|8.5 MB|
|Case sensitive:|false|

## References

https://data.world/johnsnowlabs/list-of-companies-in-nasdaq-exchanges
