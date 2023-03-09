---
layout: model
title: Map Company Tickers to Subsidiary Companies (wikipedia, en)
author: John Snow Labs
name: finmapper_wikipedia_parentcompanies_ticker
date: 2023-01-13
tags: [subsidiaries, companies, acquisitions, en, licensed]
task: Chunk Mapping
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

This models allows you to, given an extracter TICKER, retrieve all the parent / subsidiary /companies acquired and/or in the same group than it.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmapper_wikipedia_parentcompanies_ticker_en_1.0.0_3.0_1673610848433.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finmapper_wikipedia_parentcompanies_ticker_en_1.0.0_3.0_1673610848433.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetector()\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained("finner_ticker", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")\

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

CM = finance.ChunkMapperModel()\
      .pretrained('finmapper_wikipedia_parentcompanies_ticker','en','finance/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")

nlpPipeline = nlp.Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter,
        CM
])

text = ["""ABG is a multinational corporation that is engaged in ..."""]

test_data = spark.createDataFrame([text]).toDF("text")

model = nlpPipeline.fit(test_data)

lp = nlp.LightPipeline(model)
res= model.transform(test_data)
```

</div>

## Results

```bash
{'mappings': ['ABSA Group Limited',
   'ABSA Group Limited@http://www.wikidata.org/entity/Q58641733',
   'ABSA Group Limited@ABSA Group Limited@en',
   'ABSA Group Limited@http://www.wikidata.org/prop/direct/P749',
   'ABSA Group Limited@is_parent_of',
   'ABSA Group Limited@JOHANNESBURG STOCK EXCHANGE@en',
   'ABSA Group Limited@باركليز@ar',
   'ABSA Group Limited@http://www.wikidata.org/entity/Q245343'],
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finmapper_wikipedia_parentcompanies_ticker|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|1.3 MB|

## References

Wikidata
