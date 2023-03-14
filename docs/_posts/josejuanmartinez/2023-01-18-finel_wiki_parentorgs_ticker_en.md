---
layout: model
title: Resolve Company Names to Tickers using Wikidata
author: John Snow Labs
name: finel_wiki_parentorgs_ticker
date: 2023-01-18
tags: [en, licensed]
task: Entity Resolution
language: en
nav_key: models
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model helps you retrieve the TICKER of a company using a previously detected ORG entity with NER.

It also retrieves the normalized company name as per Wikidata, which can be retrieved from `aux_label` column in metadata.


## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finel_wiki_parentorgs_ticker_en_1.0.0_3.0_1674038769879.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    
resolver = finance.SentenceEntityResolverModel.pretrained("finel_wiki_parentorgs_tickers", "en", "finance/models")\
      .setInputCols(["sentence_embeddings"]) \
      .setOutputCol("normalized_name")\
      .setDistanceFunction("EUCLIDEAN")

pipelineModel = nlp.Pipeline(
      stages = [
          documentAssembler,
          embeddings,
          resolver
      ])

lp = nlp.LightPipeline(pipelineModel)
test_pred = lp.fullAnnotate('Alphabet Incorporated')
print(test_pred[0]['normalized_name'][0].result)
print(test_pred[0]['normalized_name'][0].metadata['all_k_aux_labels'].split(':::')[0])
```

</div>

## Results

```bash
GOOGL
Aux data: Alphabet Inc.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finel_wiki_parentorgs_ticker|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[original_company_name]|
|Language:|en|
|Size:|2.8 MB|
|Case sensitive:|false|

## References

Wikipedia dump about company subsidiaries
