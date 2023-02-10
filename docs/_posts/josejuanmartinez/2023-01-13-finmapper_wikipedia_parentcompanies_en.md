---
layout: model
title: Map Companies to their Acquisitions (wikipedia, en)
author: John Snow Labs
name: finmapper_wikipedia_parentcompanies
date: 2023-01-13
tags: [parent, companies, subsidiaries, en, licensed]
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

This models allows you to, given an extracter ORG, retrieve all the parent / subsidiary /companies acquired and/or in the same group than it.

IMPORTANT: This requires an exact match as the name appears in Wikidata. If you are not sure the name is the same, pleas run `finmapper_wikipedia_parentcompanies` to normalize the company name first.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmapper_wikipedia_parentcompanies_en_1.0.0_3.0_1673610612510.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finmapper_wikipedia_parentcompanies_en_1.0.0_3.0_1673610612510.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained('finner_orgs_prods_alias', 'en', 'finance/models')\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")

cm = finance.ChunkMapperModel().pretrained("finmapper_wikipedia_parentcompanies", "en", "finance/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setEnableFuzzyMatching(True)

nlpPipeline = nlp.Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter,
    cm
])

text = ["""The AES Corporation is a Fortune 500 global energy company that owns and operates regional utilities and develops renewable energy projects, including forms of clean energy generation (solar, wind, etc.) and distribution, energy battery storage, and digital energy management applications."""]

test_data = spark.createDataFrame([text]).toDF("text")

model = nlpPipeline.fit(test_data)

lp = nlp.LightPipeline(model)

lp.fullAnnotate(text)
```

</div>

## Results

```bash
{
    "uri": "http://www.wikidata.org/entity/Q291508",
    "language": "AES Corporation@en",
    "relationship": "http://www.wikidata.org/prop/direct/P355",
    "relationship_label": "is_parent_of",
    "exchange": "New York Stock Exchange@en",
    "ticker": "AES",
    "related_company": "AES Andes@en",
    "related_company_uri": "http://www.wikidata.org/entity/Q4651363"
}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finmapper_wikipedia_parentcompanies|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|852.6 KB|

## References

Wikidata
