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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmapper_nasdaq_companyname_en_1.0.0_3.2_1660038424307.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

chunk_embeddings = nlp.UniversalSentenceEncoder.pretrained("tfhub_use_lg", "en")\
    .setInputCols(["ner_chunk_doc"])\
    .setOutputCol("chunk_embeddings")

use_er_model = finance.SentenceEntityResolverModel.pretrained('finel_nasdaq_data_company_name', 'en', 'finance/models')\
    .setInputCols("chunk_embeddings")\
    .setOutputCol('normalized')\
    .setDistanceFunction("EUCLIDEAN")  
##########################################################################

CM = finance.ChunkMapperModel()\
      .pretrained('finmapper_nasdaq_companyname', 'en', 'finance/models')\
      .setInputCols(["normalized"])\ #or ner_chunk without normalization
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
                                 
text = """Altaba Inc. is a company which ..."""

test_data = spark.createDataFrame([[text]]).toDF("text")

model = pipeline.fit(test_data)

lp = nlp.LightPipeline(model)

lp.fullAnnotate(text)
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
