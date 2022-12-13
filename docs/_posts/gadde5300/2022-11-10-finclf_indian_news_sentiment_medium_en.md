---
layout: model
title: Financial Indian News Sentiment Analysis (Medium)
author: John Snow Labs
name: finclf_indian_news_sentiment_medium
date: 2022-11-10
tags: [en, finance, licensed, classification, sentiment, indian]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a `md` version of Indian News Sentiment Analysis Text Classifier, which will retrieve if a text is either expression a Positive Emotion or a Negative one.

## Predicted Entities

`POSITIVE`, `NEGATIVE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_indian_news_sentiment_medium_en_1.0.0_3.0_1668058635760.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_indian_news_sentiment_medium_en_1.0.0_3.0_1668058635760.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 

document_assembler = nlp.DocumentAssembler() \
                .setInputCol("text") \
                .setOutputCol("document")

tokenizer = nlp.Tokenizer() \
                .setInputCols(["document"]) \
                .setOutputCol("token")
      
classifierdl = finance.BertForSequenceClassification.pretrained("finclf_indian_news_sentiment_medium","en", "finance/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("label")

bert_clf_pipeline = Pipeline(stages=[document_assembler,
                                     tokenizer,
                                     classifierdl])

text = ["Eliminating shadow economy to have positive impact on GDP : Arun Jaitley"]
empty_df = spark.createDataFrame([[""]]).toDF("text")
model = bert_clf_pipeline.fit(empty_df)
res = model.transform(spark.createDataFrame([text]).toDF("text"))


```

</div>

## Results

```bash
+------------------------------------------------------------------------+----------+
|text                                                                    |result    |
+------------------------------------------------------------------------+----------+
|Eliminating shadow economy to have positive impact on GDP : Arun Jaitley|[POSITIVE]|
+------------------------------------------------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_indian_news_sentiment_medium|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|412.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

An in-house augmented version of [this dataset](https://www.kaggle.com/datasets/harshrkh/india-financial-news-headlines-sentiments)

## Benchmarking

```bash

              precision    recall  f1-score   support

    NEGATIVE       0.85      0.86      0.86     10848
    POSITIVE       0.83      0.83      0.83      9202

    accuracy                           0.84     20050
   macro avg       0.84      0.84      0.84     20050
weighted avg       0.84      0.84      0.84     20050

```
