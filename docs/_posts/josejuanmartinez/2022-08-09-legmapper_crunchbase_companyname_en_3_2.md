---
layout: model
title: Mapping Company Names to Crunchbase database
author: John Snow Labs
name: legmapper_crunchbase_companyname
date: 2022-08-09
tags: [en, legal, companies, crunchbase, data, augmentation, licensed]
task: Chunk Mapping
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model allows you to, given an extracted name of a company, get information about that company (including category / sector, country, status, initial funding, etc), as stored in Crunchbase.

It can be optionally combined with Entity Resolution to normalize first the name of the company.

This model only contains information up to 2015.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FIN_LEG_COMPANY_AUGMENTATION/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legmapper_crunchbase_companyname_en_1.0.0_3.2_1660039125941.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = legal.NerModel.pretrained('legner_orgs_prods_alias', 'en', 'legal/models')\
        .setInputCols(["document", "token", "embeddings"])\
        .setOutputCol("ner")
 
ner_converter = nlp.NerConverterInternal()\
      .setInputCols(["document", "token", "ner"])\
      .setOutputCol("ner_chunk")

# Optional: We normalize the name of the company using Crunchbase data
############################################################
chunkToDoc = nlp.Chunk2Doc()\
        .setInputCols("ner_chunk")\
        .setOutputCol("ner_chunk_doc")

chunk_embeddings = nlp.UniversalSentenceEncoder.pretrained("tfhub_use", "en") \
      .setInputCols("ner_chunk_doc") \
      .setOutputCol("sentence_embeddings")
    
resolver = legal.SentenceEntityResolverModel.pretrained("legel_crunchbase_companynames", "en", "legal/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("normalized")\
      .setDistanceFunction("EUCLIDEAN")
############################################################

CM = legal.ChunkMapperModel.pretrained("legmapper_crunchbase_companyname", "en", "legal/models")\
      .setInputCols(["normalized"])\ # or ner_chunk if you don't use normalization
      .setOutputCol("mappings")\
      .setRel('category_code')

pipeline = nlp.Pipeline().setStages([document_assembler,
                                 tokenizer, 
                                 embeddings,
                                 ner_model, 
                                 ner_converter,
                                 chunkToDoc, # Optional (normalization)
                                 chunk_embeddings, # Optional (normalization)
                                 resolver, # Optional (normalization)
                                 CM])
                                 
text = """Tokalas is a company which operates in the biopharmaceutical sector."""

test_data = spark.createDataFrame([[text]]).toDF("text")

model = pipeline.fit(test_data)
lp = nlp.LightPipeline(model)

res= lp.fullAnnotate(text)
```

</div>

## Results

```bash
[Row(mappings=[Row(annotatorType='labeled_dependency', begin=0, end=6, result='/company/tokalas', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'permalink', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='Tokalas', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'name', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'homepage_url', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='biotech', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'category_code', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='3,090,000', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'funding_total_usd', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='operating', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'status', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='USA', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'country_code', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='CA', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'state_code', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='San Diego', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'region', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='DEL MAR', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'city', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='1.0', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'funding_rounds', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='1/1/13', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'founded_at', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='2013-01', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'founded_month', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='2013-Q1', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'founded_quarter', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='2013.0', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'founded_year', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='3/5/14', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'first_funding_at', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='3/5/14', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'last_funding_at', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=6, result='', metadata={'sentence': '0', 'chunk': '0', 'entity': 'Tokalas', 'relation': 'last_milestone_at', 'all_relations': ''}, embeddings=[])])]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legmapper_crunchbase_companyname|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|6.1 MB|