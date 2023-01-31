---
layout: model
title: Extract Broker Recommendations from Broker Reports
author: John Snow Labs
name: finclf_bert_broker_recommendation
date: 2023-01-31
tags: [finance, en, licensed, bert, classification, broker_reports, tensorflow]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Text Classifier will identify whether a broker's report recommends a downgrade, maintain, upgrade, or other action.

## Predicted Entities

`Downgrade`, `Maintain`, `Upgrade`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_bert_broker_recommendation_en_1.0.0_3.0_1675164395307.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_bert_broker_recommendation_en_1.0.0_3.0_1675164395307.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
# Test classifier in Spark NLP pipeline
document_assembler = nlp.DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = nlp.Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

# Load newly trained classifier
sequenceClassifier_loaded = finance.BertForSequenceClassification.pretrained("finclf_bert_broker_recommendation", "en", "finance/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier_loaded    
])

# Generating example
example = spark.createDataFrame([["PLACE YOUR TEXT HERE"]]).toDF("text")

result = pipeline.fit(example).transform(example)

# Checking results
result.select("text", "class.result").show(truncate=False)
```

</div>

## Results

```bash
+---------+
|result   |
+---------+
|[Upgrade]|
+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_bert_broker_recommendation|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

An in-house annotated dataset of broker reports.

## Benchmarking

```bash
label          precision    recall  f1-score   support
   Downgrade       0.96      1.00      0.98        24
    Maintain       0.97      1.00      0.98        30
     Upgrade       1.00      0.94      0.97        16
       other       1.00      0.94      0.97        16
    accuracy        -         -        0.98        86
   macro-avg       0.98      0.97      0.97        86
weighted-avg       0.98      0.98      0.98        86
```