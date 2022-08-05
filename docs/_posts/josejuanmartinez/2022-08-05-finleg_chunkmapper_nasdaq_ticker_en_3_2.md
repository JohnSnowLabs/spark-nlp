---
layout: model
title: Mapping Tickers to Company Information
author: John Snow Labs
name: finleg_chunkmapper_nasdaq_ticker
date: 2022-08-05
tags: [en, finance, licensed]
task: Chunk Mapping
language: en
edition: Spark NLP for Finance 1.0.0
spark_version: 3.2
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model allows you to, given a Ticker, get information about that company, including the Company Name, the Industry and the Sector.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finleg_chunkmapper_nasdaq_ticker_en_1.0.0_3.2_1659713370724.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('document')

tokenizer = Tokenizer()\
      .setInputCols("document")\
      .setOutputCol("token")

tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_token_classifier_ticker", "en")\
  .setInputCols(["document",'token'])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["document", "token", "ner"])\
      .setOutputCol("ner_chunk")
 
ner_converter = NerConverter()\
      .setInputCols(["document", "token", "ner"])\
      .setOutputCol("ner_chunk")

CM = ChunkMapperModel()\
      .pretrained('finleg_chunkmapper_nasdaq_ticker', 'en', 'finance/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")\
      .setRel('company_name')

pipeline = Pipeline().setStages([document_assembler,
                                 tokenizer, 
                                 embeddings,
                                 ner_model, 
                                 ner_converter, 
                                 CM])

text = ["""There are some serious purchases and sales of AMZN stock today."""]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)
res= model.transform(test_data)

res.select('mappings').collect()
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
|Model Name:|finleg_chunkmapper_nasdaq_ticker|
|Type:|finance|
|Compatibility:|Spark NLP for Finance 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|210.3 KB|

## References

https://data.world/johnsnowlabs/list-of-companies-in-nasdaq-exchanges