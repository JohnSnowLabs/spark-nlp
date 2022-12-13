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

# This is a the lighter but less accurate wayto get companies. 
# Check for much more accurate models in Models Hub / Finance.
# ==========================================================
embeddings = nlp.WordEmbeddingsModel.pretrained('glove_100d') \
        .setInputCols(['document', 'token']) \
        .setOutputCol('embeddings')

ner_model = nlp.NerDLModel.pretrained("onto_100", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
# ==========================================================
 
ner_converter = nlp.NerConverter()\
      .setInputCols(["document", "token", "ner"])\
      .setOutputCol("ner_chunk")\
      .setWhiteList(["ORG"])

CM = finance.ChunkMapperModel()\
      .pretrained('finmapper_nasdaq_companyname', 'en', 'finance/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")\
      .setRel('ticker')

pipeline = Pipeline().setStages([document_assembler,
                                 tokenizer, 
                                 embeddings,
                                 ner_model, 
                                 ner_converter, 
                                 CM])
                                 
text = ["""Altaba Inc. is a company which ..."""]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)
res= model.transform(test_data)

res.select('mappings').collect()
```

</div>

## Results

```bash
[Row(mappings=[Row(annotatorType='labeled_dependency', begin=0, end=10, result='AABA', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Altaba Inc.', 'relation': 'ticker', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=10, result='Altaba Inc.', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Altaba Inc.', 'relation': 'company_name', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=10, result='Altaba', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Altaba Inc.', 'relation': 'short_name', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=10, result='Asset Management', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Altaba Inc.', 'relation': 'industry', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=10, result='Financial Services', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Altaba Inc.', 'relation': 'sector', 'all_relations': ''}, embeddings=[])])]
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
