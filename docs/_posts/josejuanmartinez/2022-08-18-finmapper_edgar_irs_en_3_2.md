---
layout: model
title: Mapping Companies IRS to Edgar Database
author: John Snow Labs
name: finmapper_edgar_irs
date: 2022-08-18
tags: [en, finance, companies, edgar, data, augmentation, irs, licensed]
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

This Chunk Mapper model allows you to, given a detected IRS with any NER model, augment it with information available in the SEC Edgar database. Some of the fields included in this Chunk Mapper are:
- Company Name
- Sector
- Former names
- Address, Phone, State
- Dates where the company submitted filings
- etc.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FIN_LEG_COMPANY_AUGMENTATION/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmapper_edgar_irs_en_1.0.0_3.2_1660817662889.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finmapper_edgar_irs_en_1.0.0_3.2_1660817662889.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = nlp.WordEmbeddingsModel.pretrained('glove_100d') \
        .setInputCols(['document', 'token']) \
        .setOutputCol('embeddings')

ner_model = nlp.NerDLModel.pretrained("onto_100", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
 
ner_converter = nlp.NerConverter()\
      .setInputCols(["document", "token", "ner"])\
      .setOutputCol("ner_chunk")\
      .setWhiteList(["CARDINAL"])

CM = finance.ChunkMapperModel()\
      .pretrained("finmapper_edgar_irs", "en", "finance/models")\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")

pipeline = Pipeline().setStages([document_assembler,
                                 tokenizer, 
                                 embeddings,
                                 ner_model, 
                                 ner_converter, 
                                 CM])

text = ["""873474341 is an American multinational corporation that is engaged in the design, development, manufacturing, and worldwide marketing and sales of footwear, apparel, equipment, accessories, and services"""]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)
res= model.transform(test_data)
```

</div>

## Results

```bash
[Row(mappings=[Row(annotatorType='labeled_dependency', begin=0, end=8, result='Masterworks 096, LLC', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'name', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='RETAIL-RETAIL STORES, NEC [5990]', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'sic', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='5990', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'sic_code', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='873474341', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'irs_number', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='1231', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'fiscal_year_end', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='NY', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'state_location', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='DE', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'state_incorporation', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='225 LIBERTY STREET', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'business_street', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='NEW YORK', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'business_city', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='NY', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'business_state', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='10281', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'business_zip', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='2035185172', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'business_phone', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'former_name', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'former_name_date', 'all_relations': ''}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='2022-01-10', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'date', 'all_relations': '2022-04-26:::2021-11-17'}, embeddings=[]), Row(annotatorType='labeled_dependency', begin=0, end=8, result='1894064', metadata={'sentence': '0', 'chunk': '0', 'entity': '873474341', 'relation': 'company_id', 'all_relations': ''}, embeddings=[])])]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finmapper_edgar_irs|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|5.7 MB|

## References

Manually scrapped Edgar Database