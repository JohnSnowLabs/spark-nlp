---
layout: model
title: Sentiment Analysis on Financial Texts
author: John Snow Labs
name: finclf_bert_news_tweets_sentiment_analysis
date: 2023-03-03
tags: [en, finance, classification, licensed, news, tweets, bert, sentiment, tensorflow]
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

This Sentiment Analysis Text Classifier has been trained on  a collection of financial news articles and tweets that have been labeled with three different classes: `Bullish`, `Bearish` and `Neutral`. The dataset on which the model has been trained on covers a wide range of financial topics including stocks, bonds, currencies, and commodities.

## Predicted Entities

`Bearish`, `Bullish`, `Neutral`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_bert_news_tweets_sentiment_analysis_en_1.0.0_3.0_1677854100676.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_bert_news_tweets_sentiment_analysis_en_1.0.0_3.0_1677854100676.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

pipeline = nlp.Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier_loaded    
])

# Generating example
example = spark.createDataFrame([['''Operating profit , excluding non-recurring items , totaled EUR 0.2 mn , down from EUR 0.8 mn in the corresponding period in 2006 .''']]).toDF("text")

result = pipeline.fit(example).transform(example)

# Checking results
result.select("text", "class.result").show(truncate=False)

```

</div>

## Results

```bash

+----------------------------------------------------------------------------------------------------------------------------------+---------+
|text                                                                                                                              |result   |
+----------------------------------------------------------------------------------------------------------------------------------+---------+
|Operating profit , excluding non-recurring items , totaled EUR 0.2 mn , down from EUR 0.8 mn in the corresponding period in 2006 .|[Bearish]|
+----------------------------------------------------------------------------------------------------------------------------------+---------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_bert_news_tweets_sentiment_analysis|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|402.4 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

In-house dataset

## Benchmarking

```bash
label              precision    recall  f1-score   support
     Bearish       0.84      0.88      0.86       487
     Bullish       0.87      0.91      0.89       872
     Neutral       0.90      0.84      0.87      1001
    accuracy         -        -        0.87      2360
   macro-avg       0.87      0.88      0.87      2360
weighted-avg       0.87      0.87      0.87      2360
```
