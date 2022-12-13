---
layout: model
title: Mapping Company Names to Crunchbase database
author: John Snow Labs
name: finmapper_crunchbase_companyname
date: 2022-08-09
tags: [en, finance, companies, crunchbase, data, augmentation, licensed]
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

This model allows you to, given an extracted name of a company, get information about that company (including category / sector, country, status, initial funding, etc), as stored in Crunchbase.

This model only contains information up to 2015.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FIN_LEG_COMPANY_AUGMENTATION/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmapper_crunchbase_companyname_en_1.0.0_3.2_1660038928665.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finmapper_crunchbase_companyname_en_1.0.0_3.2_1660038928665.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

CM = finance.ChunkMapperModel.pretrained("finmapper_crunchbase_companyname", "en", "finance/models")\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")\
      .setRel('category_code')

pipeline = Pipeline().setStages([document_assembler,
                                 tokenizer, 
                                 embeddings,
                                 ner_model, 
                                 ner_converter, 
                                 CM])
                                 
text = ["""Tokalas is a company which operates in the biopharmaceutical sector."""]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)
res= model.transform(test_data)
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
|Model Name:|finmapper_crunchbase_companyname|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|6.1 MB|
