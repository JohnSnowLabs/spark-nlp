---
layout: model
title: Financial Job Titles NER
author: John Snow Labs
name: finner_bert_roles
date: 2022-08-30
tags: [en, finance, ner, job, titles, jobs, roles, licensed]
task: Named Entity Recognition
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
recommended: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Financial nlp.BertForTokenClassification NER model aimed to extract Job Titles / Roles of people in Companies, and was trained using Resumes, Wikipedia Articles, Financial and Legal documents, annotated in-house.

## Predicted Entities

`ROLE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINNER_ROLES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_bert_roles_en_1.0.0_3.2_1661846269918.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from johnsnowlabs import *

documentAssembler = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
  .setInputCols("document")\
  .setOutputCol("token")

tokenClassifier = finance.BertForTokenClassification.pretrained("finner_bert_roles", "en", "finance/models")\
  .setInputCols("token", "document")\
  .setOutputCol("label")\
  .setCaseSensitive(True)

ner_converter = nlp.NerConverter()\
        .setInputCols(["document","token","label"])\
        .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[
  documentAssembler,
  tokenizer,
  tokenClassifier,
    ner_converter
    ]
)

import pandas as pd

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))


text = 'Jeffrey Preston Bezos is an American entrepreneur, founder and CEO of Amazon'

res = p_model.transform(spark.createDataFrame([[text]]).toDF("text"))

result_df = res.select(F.explode(F.arrays_zip(res.token.result,res.label.result, res.label.metadata)).alias("cols"))\
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("label"),
                          F.expr("cols['2']['confidence']").alias("confidence"))

result_df.show(50, truncate=100)
```

</div>

## Results

```bash
+------------+---------+----------+
|       token|ner_label|confidence|
+------------+---------+----------+
|     Jeffrey|        O|    0.9984|
|     Preston|        O|    0.9878|
|       Bezos|        O|    0.9939|
|          is|        O|     0.999|
|          an|        O|    0.9988|
|    American|   B-ROLE|    0.8294|
|entrepreneur|   I-ROLE|    0.9358|
|           ,|        O|    0.9979|
|     founder|   B-ROLE|    0.8645|
|         and|        O|     0.857|
|         CEO|   B-ROLE|      0.72|
|          of|        O|     0.995|
|      Amazon|        O|    0.9428|
+------------+---------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_bert_roles|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|402.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

In-house annotations on Wikidata, CUAD dataset, Financial 10-K documents and Resumes

## Benchmarking

```bash
label             tp     fp    fn    prec        rec          f1
B-ROLE            3553   174   262   0.95331365	 0.9313237    0.9421904
I-ROLE            4868   250   243   0.9511528	 0.95245546   0.9518037
Macro-average     8421   424   505   0.9522332   0.9418896    0.9470331
Micro-average     8421   424   505   0.9520633   0.9434237    0.94772375
```