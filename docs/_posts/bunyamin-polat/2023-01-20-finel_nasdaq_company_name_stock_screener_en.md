---
layout: model
title: Company Name Normalization using Nasdaq Stock Screener
author: John Snow Labs
name: finel_nasdaq_company_name_stock_screener
date: 2023-01-20
tags: [en, finance, licensed, nasdaq, company]
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

This is a Financial Entity Resolver model, trained to obtain normalized versions of Company Names, registered in NASDAQ Stock Screener. You can use this model after extracting a company name using any NER, and you will obtain the official name of the company as per NASDAQ Stock Screener.

After this, you can use `finmapper_nasdaq_company_name_stock_screener` to augment and obtain more information about a company using NASDAQ Stock Screener, including Ticker, Sector, Country, etc.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finel_nasdaq_company_name_stock_screener_en_1.0.0_3.0_1674233034536.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finel_nasdaq_company_name_stock_screener_en_1.0.0_3.0_1674233034536.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained("finner_orgs_prods_alias", "en", "finance/models")\
    .setInputCols(["document", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["document","token","ner"])\
    .setOutputCol("ner_chunk")

chunkToDoc = nlp.Chunk2Doc()\
    .setInputCols("ner_chunk")\
    .setOutputCol("ner_chunk_doc")

chunk_embeddings = nlp.UniversalSentenceEncoder.pretrained("tfhub_use", "en") \
    .setInputCols("ner_chunk_doc") \
    .setOutputCol("sentence_embeddings")

use_er_model = finance.SentenceEntityResolverModel.pretrained("finel_nasdaq_company_name_stock_screener", "en", "finance/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("normalized")\
    .setDistanceFunction("EUCLIDEAN")

nlpPipeline = nlp.Pipeline(stages=[
     documentAssembler,
     tokenizer,
     embeddings,
     ner_model,
     ner_converter,
     chunkToDoc,
     chunk_embeddings,
     use_er_model
])

text = """NIKE is an American multinational corporation that is engaged in the design, development, manufacturing, and worldwide marketing and sales of footwear, apparel, equipment, accessories, and services."""

test_data = spark.createDataFrame([[text]]).toDF("text")

model = nlpPipeline.fit(test_data)

lp = nlp.LightPipeline(model)

result = lp.annotate(text)

result["normalized"]
```

</div>

## Results

```bash
['Nike Inc. Common Stock']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finel_nasdaq_company_name_stock_screener|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[normalized]|
|Language:|en|
|Size:|54.7 MB|
|Case sensitive:|false|

## References

https://www.nasdaq.com/market-activity/stocks/screener
