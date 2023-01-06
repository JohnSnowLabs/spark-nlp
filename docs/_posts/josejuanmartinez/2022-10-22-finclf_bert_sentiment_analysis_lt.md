---
layout: model
title: Financial Sentiment Analysis (Lithuanian)
author: John Snow Labs
name: finclf_bert_sentiment_analysis
date: 2022-10-22
tags: [lt, legal, classification, sentiment, analysis, licensed]
task: Text Classification
language: lt
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Lithuanian Sentiment Analysis Text Classifier, which will retrieve if a text is either expression a Positive Emotion or a Negative one.

## Predicted Entities

`POS`,`NEG`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_bert_sentiment_analysis_lt_1.0.0_3.0_1666475378253.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
sequenceClassifier_loaded = finance.BertForSequenceClassification.pretrained("finclf_bert_sentiment_analysis", "lt", "finance/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier_loaded    
])

# Generating example
example = spark.createDataFrame([["Pagalbos paraðiuto laukiantis verslas priemones vertina teigiamai  tik yra keli „jeigu“"]]).toDF("text")

result = pipeline.fit(example).transform(example)

# Checking results
result.select("text", "class.result").show(truncate=False)
```

</div>

## Results

```bash
+---------------------------------------------------------------------------------------+------+
|text                                                                                   |result|
+---------------------------------------------------------------------------------------+------+
|Pagalbos paraðiuto laukiantis verslas priemones vertina teigiamai  tik yra keli „jeigu“|[POS] |
+---------------------------------------------------------------------------------------+------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_bert_sentiment_analysis|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|lt|
|Size:|406.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

An in-house augmented version of [this dataset](https://www.kaggle.com/datasets/rokastrimaitis/lithuanian-financial-news-dataset-and-bigrams?select=dataset%28original%29.csv) removing NEU tag

## Benchmarking

```bash
       label    precision    recall  f1-score   support
         NEG       0.80      0.76      0.78       509
         POS       0.90      0.92      0.91      1167
    accuracy         -         -       0.87      1676
   macro-avg       0.85      0.84      0.84      1676
weighted-avg       0.87      0.87      0.87      1676
```
