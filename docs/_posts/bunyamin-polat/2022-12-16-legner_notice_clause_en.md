---
layout: model
title: Notice Clause NER Model
author: John Snow Labs
name: legner_notice_clause
date: 2022-12-16
tags: [en, legal, ner, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is an NER model aimed to be used in notice clauses, to retrieve entities as NOTICE_METHOD, NOTICE_PARTY, ADDRESS, EMAIL, etc. Make sure you run this model only on notice clauses, after you filter them using `legclf_notice_clause`

## Predicted Entities

`ADDRESS`, `DEPARTMENT`, `EMAIL`, `FAX`, `NAME`, `NOTICE_METHOD`, `NOTICE_PARTY`, `PERSON`, `PHONE`, `TITLE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_notice_clause_en_1.0.0_3.0_1671211179919.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained("legner_notice_clause", "en", "legal/models") \
    .setInputCols(["document", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = nlp.NerConverter() \
    .setInputCols(["document","token","ner"]) \
    .setOutputCol("ner_chunk")

pipeline = nlp.Pipeline(stages=[
    document_assembler, 
    tokenizer,
    embeddings,
    ner_model,
    ner_converter   
    ])

empty_df = spark.createDataFrame([['']]).toDF("text")

ner_model = pipeline.fit(empty_df)

data = spark.createDataFrame([["""Source: FUELCELL ENERGY INC, 8-K, 11/6/2019
ExxonMobil: ExxonMobil Research and Engineering Company 1545 Route 22 East Annandale, NJ 08801-0900 Attention: Timothy Barckholtz, Senior Scientific Advisor Email: tim.barckholtz@exxonmobil.com FCE: FuelCell Energy, Inc. 782"""]]).toDF("text")

result = ner_model.transform(data)
```

</div>

## Results

```bash
+---------------------------------------------------------------------+------------+
|ner_chunk                                                            |label       |
+---------------------------------------------------------------------+------------+
|ExxonMobil                                                           |NOTICE_PARTY|
|ExxonMobil Research and Engineering Company                          |NAME        |
|1545 Route 22 East Annandale, NJ 08801-0900                          |ADDRESS     |
|Timothy Barckholtz                                                   |PERSON      |
|Senior Scientific Advisor                                            |TITLE       |
|tim.barckholtz@exxonmobil.com                                        |EMAIL       |
+---------------------------------------------------------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_notice_clause|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|1.1 MB|

## References

In-house dataset

## Benchmarking

```bash
        label     precision  recall  f1-score   support
      ADDRESS       0.86      0.94      0.90       141
   DEPARTMENT       0.75      0.27      0.40        11
        EMAIL       0.92      1.00      0.96        48
          FAX       0.65      0.88      0.75        51
         NAME       0.78      0.79      0.79       140
NOTICE_METHOD       0.74      0.80      0.77       353
 NOTICE_PARTY       0.77      0.85      0.81       103
       PERSON       0.91      0.94      0.92       114
        PHONE       0.60      0.47      0.53        19
        TITLE       0.76      0.90      0.82        80
    micro-avg       0.78      0.85      0.81      1060
    macro-avg       0.77      0.79      0.77      1060
 weighted-avg       0.79      0.85      0.81      1060
```
