---
layout: model
title: Augment Company Names with NASDAQ database
author: John Snow Labs
name: finmapper_nasdaq_companyname
date: 2022-08-09
tags: [en, finance, companies, tickers, nasdaq, data, augmentation, licensed]
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

This model allows you to, given an extracted name of a company, get information about that company, including the Industry, the Sector and the Trading Symbol (ticker).

It can be optionally combined with Entity Resolution to normalize first the name of the company.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FIN_LEG_COMPANY_AUGMENTATION/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmapper_nasdaq_companyname_en_1.0.0_3.2_1660038424307.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finmapper_nasdaq_companyname_en_1.0.0_3.2_1660038424307.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    .setInputCols(["document", "token"])\
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained('finner_orgs_prods_alias', 'en', 'finance/models')\
    .setInputCols(["document", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["document", "token", "ner"])\
    .setOutputCol("ner_chunk")

CM = finance.ChunkMapperModel().pretrained('finmapper_nasdaq_companyname', 'en', 'finance/models')\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setEnableFuzzyMatching(True)

pipeline = nlp.Pipeline().setStages([document_assembler,
                          tokenizer, 
                          embeddings,
                          ner_model, 
                          ner_converter,
                          CM])
                          
text = """Altaba Inc. is a company which ..."""

test_data = spark.createDataFrame([[text]]).toDF("text")

model = pipeline.fit(test_data)

lp = nlp.LightPipeline(model)

lp.fullAnnotate(text)
```

</div>

## Results

```bash
{
    "ticker": "AABA",
    "company_name": "Altaba Inc.",
    "short_name": "Altaba",
    "industry": "Asset Management",
    "sector": "Financial Services"
}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finmapper_nasdaq_companyname|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|210.5 KB|

## References

https://data.world/johnsnowlabs/list-of-companies-in-nasdaq-exchanges
