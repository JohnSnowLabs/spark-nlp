---
layout: model
title: Mapping Companies to NASDAQ Stock Screener by Company Name
author: John Snow Labs
name: finmapper_nasdaq_company_name_stock_screener
date: 2023-01-19
tags: [en, finance, licensed, nasdaq, company]
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

This model allows you to, given an extracted name of a company, get following information about that company from Nasdaq Stock Screener:

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

It can be optionally combined with Entity Resolution to normalize first the name of the company.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmapper_nasdaq_company_name_stock_screener_en_1.0.0_3.0_1674161310624.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finmapper_nasdaq_company_name_stock_screener_en_1.0.0_3.0_1674161310624.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = finance.NerModel.pretrained('finner_orgs_prods_alias', 'en', 'finance/models')\
    .setInputCols(["document", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["document", "token", "ner"])\
    .setOutputCol("ner_chunk")

# Optional: To normalize the ORG name using NASDAQ data before the mapping
##########################################################################
chunkToDoc = nlp.Chunk2Doc()\
    .setInputCols("ner_chunk")\
    .setOutputCol("ner_chunk_doc")

chunk_embeddings = nlp.UniversalSentenceEncoder.pretrained("tfhub_use", "en")\
    .setInputCols(["ner_chunk_doc"])\
    .setOutputCol("chunk_embeddings")

use_er_model = finance.SentenceEntityResolverModel.pretrained('finel_nasdaq_company_name_stock_screener', 'en', 'finance/models')\
    .setInputCols("chunk_embeddings")\
    .setOutputCol('normalized')\
    .setDistanceFunction("EUCLIDEAN")  
##########################################################################

CM = finance.ChunkMapperModel.pretrained('finmapper_nasdaq_company_name_stock_screener', 'en', 'finance/models')\
    .setInputCols(["normalized"])\
    .setOutputCol("mappings")

pipeline = nlp.Pipeline().setStages([document_assembler,
                                 tokenizer, 
                                 embeddings,
                                 ner_model, 
                                 ner_converter,
                                 chunkToDoc, # Optional for normalization
                                 chunk_embeddings, # Optional for normalization
                                 use_er_model, # Optional for normalization
                                 CM])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

lp = nlp.LightPipeline(model)

text = """Nike is an American multinational association that is involved in the design, development, manufacturing and worldwide marketing and sales of apparel, footwear, accessories, equipment and services."""

result = lp.fullAnnotate(text)
```

</div>

## Results

```bash
"Country": "United States",
"IPO_Year": "0",
"Industry": "Shoe Manufacturing",
"Last_Sale": "$128.85",
"Market_Cap": "1.9979004036E11",
"Name": "Nike Inc. Common Stock",
"Net_Change": "0.96",
"Percent_Change": "0.751%",
"Sector": "Consumer Discretionary",
"Symbol": "NKE",
"Volume": "4854668"
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finmapper_nasdaq_company_name_stock_screener|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|599.1 KB|

## References

https://www.nasdaq.com/market-activity/stocks/screener
