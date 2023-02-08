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

CM = finance.ChunkMapperModel().pretrained("finmapper_edgar_irs", "en", "finance/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setEnableFuzzyMatching(True)

pipeline = nlp.Pipeline().setStages([document_assembler,
                          tokenizer, 
                          embeddings,
                          ner_model, 
                          ner_converter, 
                          CM])

text = ["""873474341 is an American multinational corporation that is engaged in the design, development, manufacturing, and worldwide marketing and sales of footwear, apparel, equipment, accessories, and services"""]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)

res = model.transform(test_data)
```

</div>

## Results

```bash
{
    "name": "Masterworks 096, LLC",
    "sic": "RETAIL-RETAIL STORES, NEC [5990]",
    "sic_code": "5990",
    "irs_number": "873474341",
    "fiscal_year_end": "1231",
    "state_location": "NY",
    "state_incorporation": "DE",
    "business_street": "225 LIBERTY STREET",
    "business_city": "NEW YORK",
    "business_state": "NY",
    "business_zip": "10281",
    "business_phone": "2035185172",
    "former_name": "",
    "former_name_date": "",
    "date": "2022-01-10",
    "company_id": "1894064"
}
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