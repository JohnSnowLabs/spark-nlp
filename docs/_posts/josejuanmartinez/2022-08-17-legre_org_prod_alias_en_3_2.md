---
layout: model
title: Legal Relation Extraction (Alias)
author: John Snow Labs
name: legre_org_prod_alias
date: 2022-08-17
tags: [en, legal, re, relations, licensed]
task: Relation Extraction
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model can be used to extract Aliases of Companies or Product names. An "Alias" is a named used in a document to refer to the original name of a company or product. Examples:

- John Snow Labs, also known as JSL
- John Snow Labs ("JSL")
- etc

## Predicted Entities

`has_alias`, `has_collective_alias`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_org_prod_alias_en_1.0.0_3.2_1660739037434.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legre_org_prod_alias_en_1.0.0_3.2_1660739037434.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["document"])\
        .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
        .setInputCols(["document", "token"]) \
        .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained("legner_orgs_prods_alias", "en", "legal/models")\
        .setInputCols(["document", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["document","token","ner"])\
        .setOutputCol("ner_chunk")

reDL = legal.RelationExtractionDLModel()\
    .pretrained("legre_org_prod_alias", "en", "legal/models")\
    .setPredictionThreshold(0.1)\
    .setInputCols(["ner_chunk", "document"])\
    .setOutputCol("relations")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter,
        reDL])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text='''
On March 12, 2020 we closed a Loan and Security Agreement with Hitachi Capital America Corp. ("Hitachi") the terms of which are described in this report which replaced our credit facility with Western Alliance Bank.
'''

lmodel = LightPipeline(model)
lmodel.fullAnnotate(text)
```

</div>

## Results

```bash
relation	entity1	entity1_begin	entity1_end	chunk1	entity2	entity2_begin	entity2_end	chunk2	confidence
has_alias	ORG	64	92	Hitachi Capital America Corp.	ALIAS	96	102	Hitachi	0.9983972
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_org_prod_alias|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|409.9 MB|

## References

Manual annotations on CUAD dataset and 10K filings

## Benchmarking

```bash
label                    Recall    Precision   F1       Support
has_alias                0.920     1.000       0.958    50
has_collective_alias     1.000     0.750       0.857     6
no_rel                   1.000     0.957       0.978    44
Avg.                     0.973     0.902       0.931     -
Weighted-Avg.            0.960     0.966       0.961     -
```