---
layout: model
title: Mapping Company Names to Edgar Database
author: John Snow Labs
name: finmapper_edgar_companyname
date: 2022-08-18
tags: [en, finance, companies, edgar, data, augmentation, licensed]
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

This Chunk Mapper model allows you to, given a detected Organization with any NER model, augment it with information available in the SEC Edgar database. Some of the fields included in this Chunk Mapper are:
- IRS number
- Sector
- Former names
- Address, Phone, State
- Dates where the company submitted filings
- etc.

IMPORTANT: Chunk Mappers work with exact matches, so before using Chunk Mapping, you need to carry out Company Name Normalization to get how the company name is stored in Edgar. To do this, use Entity Linking, more especifically the `finel_edgar_companynames` model, with the Organization Name extracted by any NER model. You will get  the normalized version (by Edgar standards) of the name, which you can send to this model for data augmentation.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FIN_LEG_COMPANY_AUGMENTATION/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmapper_edgar_companyname_en_1.0.0_3.2_1660817326595.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finmapper_edgar_companyname_en_1.0.0_3.2_1660817326595.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
      .setInputCols(["document"])\
      .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
      .setInputCols(["document", "token"]) \
      .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained("finner_orgs_prods_alias", "en", "finance/models")\
      .setInputCols(["document", "token", "embeddings"])\
      .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
      .setInputCols(["document","token","ner"])\
      .setOutputCol("ner_chunk")

cm = finance.ChunkMapperModel().pretrained("finmapper_edgar_companyname", "en", "finance/models")\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")\
      .setEnableFuzzyMatching(True)

nlpPipeline = nlp.Pipeline(stages=[
      document_assembler,
      tokenizer,
      embeddings,
      ner_model,
      ner_converter,
      cm
])

text = """NIKE Inc is an American multinational corporation that is engaged in the design, development, manufacturing, and worldwide marketing and sales of footwear, apparel, equipment, accessories, and services"""

test_data = spark.createDataFrame([[text]]).toDF("text")

model = nlpPipeline.fit(test_data)

lp = nlp.LightPipeline(model)

result = lp.fullAnnotate(text)
```

</div>

## Results

```bash
{
    "name": "NIKE, Inc.",
    "sic": "RUBBER & PLASTICS FOOTWEAR [3021]",
    "sic_code": "3021",
    "irs_number": "930584541",
    "fiscal_year_end": "531",
    "state_location": "OR",
    "state_incorporation": "OR",
    "business_street": "ONE BOWERMAN DR",
    "business_city": "BEAVERTON",
    "business_state": "OR",
    "business_zip": "97005-6453",
    "business_phone": "5036713173",
    "former_name": "NIKE INC",
    "former_name_date": "19920703",
    "date": "2022-01-06",
    "company_id": "320187"
}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finmapper_edgar_companyname|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|11.0 MB|

## References

Manually scrapped Edgar Database
