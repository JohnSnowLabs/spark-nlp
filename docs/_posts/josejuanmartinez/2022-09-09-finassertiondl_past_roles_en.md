---
layout: model
title: Identify Job Experiences in the Past
author: John Snow Labs
name: finassertiondl_past_roles
date: 2022-09-09
tags: [en, finance, assertion, status, job, experiences, past, licensed]
task: Assertion Status
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: AssertionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is aimed to detect if any Role, Job Title, Person, Organization, Date, etc. entity, extracted with NER, is expressed as a Past Experience.

## Predicted Entities

`NO_PAST`, `PAST`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/ASSERTIONDL_PAST_ROLES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finassertiondl_past_roles_en_1.0.0_3.2_1662762393161.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finassertiondl_past_roles_en_1.0.0_3.2_1662762393161.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

# nlp.Tokenizer splits words in a relevant format for NLP
tokenizer = nlp.Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
# Add as many NER as you wish here. We have added 2 as an example.
# ================
tokenClassifier = finance.BertForTokenClassification.pretrained("finner_bert_roles", "en", "finance/models")\
  .setInputCols("token", "document")\
  .setOutputCol("label")

ner = finance.NerModel.pretrained("finner_org_per_role_date", "en", "finance/models")\
  .setInputCols("document", "token", "embeddings")\
  .setOutputCol("label2")

ner_converter = finance.NerConverterInternal() \
    .setInputCols(["document", "token", "label"]) \
    .setOutputCol("ner_chunk")

ner_converter2 = finance.NerConverterInternal() \
    .setInputCols(["document", "token", "label2"]) \
    .setOutputCol("ner_chunk2")

merger =  finance.ChunkMergeApproach()\
    .setInputCols(["ner_chunk", "ner_chunk2"])\
    .setOutputCol("merged_chunk")
# ================

assertion = finance.AssertionDLModel.pretrained("finassertiondl_past_roles", "en", "finance/models")\
    .setInputCols(["document", "merged_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    tokenizer,
    embeddings,
    tokenClassifier,
    ner,
    ner_converter,
    ner_converter2,
    merger,
    assertion
    ])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
lp = LightPipeline(model)
r = lp.fullAnnotate("Mrs. Charles was before Managing Director at Liberty, LC")
```

</div>

## Results

```bash
chunk,begin,end,entity_type,assertion
Mrs. Charles,0,11,PERSON,PAST
Managing Director,24,40,ROLE,PAST
Liberty, LC,45,55,ORG,PAST
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finassertiondl_past_roles|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, doc_chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|
|Size:|2.2 MB|

## References

In-house annotations from 10K Filings and Wikidata

## Benchmarking

```bash
label          tp    fp    fn    prec        rec          f1
NO_PAST        362   6     13    0.9836956   0.96533334   0.974428
PAST           196   13    6     0.9377990   0.97029704   0.953771
Macro-average  558   19    19    0.9607473   0.96781516   0.964268
Micro-average  558   19    19    0.9670710   0.96707106   0.967071
```
