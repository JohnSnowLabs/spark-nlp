---
layout: model
title: Sentiment Analysis on Broker's Reports
author: John Snow Labs
name: finclf_bert_broker_sentiment_analysis
date: 2023-01-31
tags: [licensed, en, finance, bert, classification, tensorflow]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: FinanceBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This English Sentiment Analysis Text Classifier will determine from a Broker's report whether a text is Positive, Negative, Neutral, or other expression.

## Predicted Entities

`Positive`, `Negitive`, `Neutral`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_bert_broker_sentiment_analysis_en_1.0.0_3.0_1675177527227.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_bert_broker_sentiment_analysis_en_1.0.0_3.0_1675177527227.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
sequenceClassifier_loaded = finance.BertForSequenceClassification.pretrained("finclf_bert_broker_sentiment_analysis", "en", "finance/models")\
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
+----------+
|result    |
+----------+
|[Negative]|
+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_bert_broker_sentiment_analysis|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|402.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

An in-house annotated dataset

## Benchmarking

```bash
label          precision    recall  f1-score   support
    Negative       1.00      0.81      0.90        16
     Neutral       0.84      0.84      0.84        25
    Positive       0.74      0.88      0.80        32
       other       1.00      0.77      0.87        13
    accuracy        -          -       0.84        86
   macro-avg       0.89      0.82      0.85        86
weighted-avg       0.86      0.84      0.84        86
```