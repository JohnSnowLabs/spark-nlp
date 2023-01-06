---
layout: model
title: Augment Company Tickers with NASDAQ database
author: John Snow Labs
name: finmapper_nasdaq_data_ticker
date: 2022-10-22
tags: [en, finance, companies, nasdaq, ticker, licensed]
task: Chunk Mapping
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Financial Chunk Mapper, which will retrieve, given a normalized Company Name (using, for example, `finer_nasdaq_data` to obtain the official Nasdaq company name), extra information about the company, including:
- Ticker
- Stock Exchange
- Section
- Sic codes
- Section
- Industry
- Category
- Currency
- Location
- Previous names (first_name)
- Company type (INC, CORP, etc)
- and some more.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FIN_LEG_COMPANY_AUGMENTATION/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmapper_nasdaq_data_ticker_en_1.0.0_3.0_1666474260714.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_converter = nlp.NerConverterInternal()\
    .setInputCols(["document", "token", "ner"])\
    .setOutputCol("ner_chunk")

CM = finance.ChunkMapperModel.pretrained('finmapper_nasdaq_data_ticker', 'en', 'finance/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")\
      .setRel('company_name')

pipeline = Pipeline().setStages([document_assembler,
                                 tokenizer, 
                                 embeddings,
                                 ner_model, 
                                 ner_converter, 
                                 CM])
                                 
text = ["""There are some serious purchases and sales of GLE1 stock today."""]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)

res= model.transform(test_data).select('mappings').collect()
```

</div>

## Results

```bash
[Row(mappings=[Row(annotatorType='labeled_dependency', begin=46, end=49, result='AMZN', metadata={'sentence': '0', 'chunk': '0', 'entity': 'AMZN', 'relation': 'ticker', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=46, end=49, result='Amazon.com Inc.', metadata={'sentence': '0', 'chunk': '0', 'entity': 'AMZN', 'relation': 'company_name', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=46, end=49, result='Amazon.com', metadata={'sentence': '0', 'chunk': '0', 'entity': 'AMZN', 'relation': 'short_name', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=46, end=49, result='Retail - Apparel & Specialty', metadata={'sentence': '0', 'chunk': '0', 'entity': 'AMZN', 'relation': 'industry', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=46, end=49, result='Consumer Cyclical', metadata={'sentence': '0', 'chunk': '0', 'entity': 'AMZN', 'relation': 'sector', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=57, end=61, result='NONE', metadata={'sentence': '0', 'chunk': '1', 'entity': 'today'}, embeddings=[])])]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finmapper_nasdaq_data_ticker|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|1.0 MB|

## References

NASDAQ Database