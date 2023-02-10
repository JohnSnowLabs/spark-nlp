---
layout: model
title: Legal NER Obligations on Agreements
author: John Snow Labs
name: legner_obligations
date: 2022-08-22
tags: [en, legal, ner, obligations, agreements, licensed]
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

IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_cuad_obligations_clause` Text Classifier to select only these paragraphs; 

This Name Entity Recognition model is aimed to extract what the different parties of an agreement commit to do. We call it "obligations", but could also be called "commitments" or "agreemeents".

This model extracts the subject (who commits to doing what), the action (the verb - will provide, shall sign...) and the object (what subject will provide, what subject shall sign, etc). Also, if the recipient of the obligation is a third party (a subject will provide to the Company X ...), then that third party (Company X) will be extracted as an indirect object.

This model also has a Relation Extraction model which can be used to connect the entities together.

The object is usually very diverse (will provide with technology? documents? people? items? etc) and often times, very long clauses. For that, we include a more advanced way to extract objects, using Automatic Question Generation (what will [subject] [action]? Example - What will the Company provide?) and Question Answering (using that question and a context, we retrieve the answer from the text). Please, check the Question Answering notebook in the Spark NLP Workshop for more information about this approach.

## Predicted Entities

`OBLIGATION_SUBJECT`, `OBLIGATION_ACTION`, `OBLIGATION`, `OBLIGATION_INDIRECT_OBJECT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_obligations_en_1.0.0_3.2_1661182145726.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_obligations_en_1.0.0_3.2_1661182145726.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from johnsnowlabs import *

documentAssembler = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sparktokenizer = nlp.Tokenizer()\
  .setInputCols("document")\
  .setOutputCol("token")


tokenClassifier = legal.BertForTokenClassification.pretrained("legner_obligations", "en", "legal/models")\
  .setInputCols("token", "document")\
  .setOutputCol("label")\
  .setCaseSensitive(True)
  
pipeline =  Pipeline(stages=[
  documentAssembler,
  sparktokenizer,
  tokenClassifier
    ]
)

import pandas as pd

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

text = """The Buyer shall use such materials and supplies only in accordance with the present agreement"""
res = p_model.transform(spark.createDataFrame([[text]]).toDF("text"))

```

</div>

## Results

```bash
+----------+--------------------+
|     token|           ner_label|
+----------+--------------------+
|       The|                   O|
|     Buyer|B-OBLIGATION_SUBJECT|
|     shall| B-OBLIGATION_ACTION|
|       use| I-OBLIGATION_ACTION|
|      such|        B-OBLIGATION|
| materials|        I-OBLIGATION|
|       and|        I-OBLIGATION|
|  supplies|        I-OBLIGATION|
|      only|        I-OBLIGATION|
|        in|        I-OBLIGATION|
|accordance|        I-OBLIGATION|
|      with|        I-OBLIGATION|
|       the|        I-OBLIGATION|
|   present|        I-OBLIGATION|
| agreement|        I-OBLIGATION|
+----------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_obligations|
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

In-house annotated documents on CUAD dataset

## Benchmarking

```bash
                       label  precision    recall  f1-score   support
                B-OBLIGATION       0.61      0.44      0.51        93
         B-OBLIGATION_ACTION       0.88      0.89      0.89        85
B-OBLIGATION_INDIRECT_OBJECT       0.69      0.71      0.70        34
        B-OBLIGATION_SUBJECT       0.80      0.87      0.84        87
                I-OBLIGATION       0.72      0.77      0.75      1251
         I-OBLIGATION_ACTION       0.80      0.79      0.79       167
        I-OBLIGATION_SUBJECT       0.75      0.43      0.55        14
                           O       0.87      0.84      0.85      2395
                    accuracy         -         -       0.81      4126
                   macro-avg       0.76      0.72      0.73      4126
                weighted-avg       0.81      0.81      0.81      4126
```