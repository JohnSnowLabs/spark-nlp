---
layout: model
title: Legal NER - License / Permission Clauses (Bert, sm)
author: John Snow Labs
name: legner_bert_grants
date: 2022-08-12
tags: [en, legal, ner, grants, permissions, licenses, licensed]
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

This model aims to detect License grants / permissions in agreements, provided by a Subject (PERMISSION_SUBJECT) to a Recipient (PERMISSION_INDIRECT_OBJECT). THe permission itself is in PERMISSION tag.

There is a lighter (non-transformer based) version of this model available as `legner_grants_md`. 

## Predicted Entities

`PERMISSION`, `PERMISSION_SUBJECT`, `PERMISSION_OBJECT`, `PERMISSION_INDIRECT_OBJECT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_bert_grants_en_1.0.0_3.2_1660292396316.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_bert_grants_en_1.0.0_3.2_1660292396316.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
  .setInputCols("document")\
  .setOutputCol("token")

tokenClassifier = legal.BertForTokenClassification.pretrained("legner_bert_grants", "en", "legal/models")\
  .setInputCols("token", "document")\
  .setOutputCol("label")\
  .setCaseSensitive(True)

pipeline =  nlp.Pipeline(stages=[
  documentAssembler,
  tokenizer,
  tokenClassifier
    ]
)

import pandas as pd

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

text = """Fox grants to Licensee a limited, exclusive (except as otherwise may be provided in this Agreement), 
non-transferable (except as permitted in Paragraph 17(d)) right and license"""

res = p_model.transform(spark.createDataFrame([[text]]).toDF("text"))

from pyspark.sql import functions as F

res.select(F.explode(F.arrays_zip('token.result', 'label.result')).alias("cols")) \
               .select(F.expr("cols['0']").alias("token"),
                       F.expr("cols['1']").alias("ner_label"))\
               .show(20, truncate=100)
```

</div>

## Results

```bash
+----------------+----------------------------+
|           token|                   ner_label|
+----------------+----------------------------+
|             Fox|        B-PERMISSION_SUBJECT|
|          grants|                           O|
|              to|                           O|
|        Licensee|B-PERMISSION_INDIRECT_OBJECT|
|               a|                           O|
|         limited|                B-PERMISSION|
|               ,|                I-PERMISSION|
|       exclusive|                I-PERMISSION|
|               (|                I-PERMISSION|
|          except|                I-PERMISSION|
|              as|                I-PERMISSION|
|       otherwise|                I-PERMISSION|
|             may|                I-PERMISSION|
|              be|                I-PERMISSION|
|        provided|                I-PERMISSION|
|              in|                I-PERMISSION|
|            this|                I-PERMISSION|
|       Agreement|                I-PERMISSION|
|              ),|                I-PERMISSION|
|non-transferable|                I-PERMISSION|
+----------------+----------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_bert_grants|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|412.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
                       label  precision    recall  f1-score   support
                B-PERMISSION       0.88      0.79      0.83        38
B-PERMISSION_INDIRECT_OBJECT       0.85      0.94      0.89        36
        B-PERMISSION_SUBJECT       0.89      0.85      0.87        40
                I-PERMISSION       0.80      0.69      0.74       342
                           O       0.94      0.97      0.95      1827
                    accuracy         -         -       0.92      2292
                   macro-avg       0.85      0.81      0.86      2292
                weighted-avg       0.91      0.92      0.91      2292
```
