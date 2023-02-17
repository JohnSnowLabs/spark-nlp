---
layout: model
title: Augment Tickers with NASDAQ database
author: John Snow Labs
name: finmapper_nasdaq_ticker
date: 2022-08-09
tags: [en, finance, companies, tickers, nasdaq, licensed]
task: Chunk Mapping
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model allows you to, given a Ticker, get information about that company, including the Company Name, the Industry and the Sector.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FIN_LEG_COMPANY_AUGMENTATION/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmapper_nasdaq_ticker_en_1.0.0_3.2_1660038524908.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finmapper_nasdaq_ticker_en_1.0.0_3.2_1660038524908.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = nlp.RoBertaForTokenClassification.pretrained("finner_roberta_ticker", "en", "finance/models")\
      .setInputCols(["document",'token'])\
      .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
      .setInputCols(["document", "token", "ner"])\
      .setOutputCol("ner_chunk")

CM = finance.ChunkMapperModel().pretrained('finmapper_nasdaq_ticker', 'en', 'finance/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")\
      .setRel('company_name')\
      .setEnableFuzzyMatching(True)

pipeline = nlp.Pipeline().setStages(
      [
          document_assembler,
          tokenizer, 
          tokenClassifier,
          ner_converter,  
          CM
      ]
      )

text = ["""There are some serious purchases and sales of AMZN stock today."""]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)

res = model.transform(test_data)

res.select('mappings').collect()
```

</div>

## Results

```bash
{
    "ticker": "AMZN",
    "company_name": "Amazon.com Inc.",
    "short_name": "Amazon.com",
    "industry": "Retail - Apparel & Specialty",
    "sector": "Consumer Cyclical"
}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finmapper_nasdaq_ticker|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|210.3 KB|

## References

https://data.world/johnsnowlabs/list-of-companies-in-nasdaq-exchanges
