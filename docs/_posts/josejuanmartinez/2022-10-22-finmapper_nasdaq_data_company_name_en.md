---
layout: model
title: Augment Company Names with NASDAQ database
author: John Snow Labs
name: finmapper_nasdaq_data_company_name
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

This is a Financial Chunk Mapper which will retrieve, given a ticker, extra information about the company, including:
- Company Name
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmapper_nasdaq_data_company_name_en_1.0.0_3.0_1666474142842.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finmapper_nasdaq_data_company_name_en_1.0.0_3.0_1666474142842.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = finance.NerModel.pretrained("finner_orgs_prods_alias","en","finance/models")\
       .setInputCols(["document", "token", "embeddings"])\
       .setOutputCol("ner")
 
ner_converter = nlp.NerConverter()\
      .setInputCols(["document", "token", "ner"])\
      .setOutputCol("ner_chunk")\
      .setWhiteList(["ORG"])

...

# Use `finel_nasdaq_data_company_name` Entity Resolver to normalize the company name
# to be able to match with Chunk Mapper

...

CM = finance.ChunkMapperModel.pretrained('finmapper_nasdaq_data_company_name', 'en', 'finance/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")\
      .setRel('ticker')

pipeline = Pipeline().setStages([document_assembler,
                                 tokenizer, 
                                 embeddings,
                                 ner_model, 
                                 ner_converter, 
                                 CM])

text = ["""GLEASON CORP is a company which ..."""]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)
res= model.transform(test_data)
```

</div>

## Results

```bash
Row(annotatorType='labeled_dependency', begin=0, end=11, relation='ticker', result='GLE1'...)
Row(annotatorType='labeled_dependency', begin=0, end=11, relation='name', result='GLEASON CORP'...)
Row(annotatorType='labeled_dependency', begin=0, end=11, relation='exchange', result='NYSE'...)
Row(annotatorType='labeled_dependency', begin=0, end=11, relation='category' result='Domestic Common Stock'...)
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finmapper_nasdaq_data_company_name|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|989.1 KB|

## References

NASDAQ Database
