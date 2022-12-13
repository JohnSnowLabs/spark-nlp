---
layout: model
title: Financial Indian News Sentiment Analysis (Small)
author: John Snow Labs
name: finclf_indian_news_sentiment
date: 2022-11-10
tags: [en, finance, licensed, classification, sentiment, indian]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a `sm` version of Indian News Sentiment Analysis Text Classifier, which will retrieve if a text is either expression a Positive Emotion or a Negative one.

## Predicted Entities

`POSITIVE`, `NEGATIVE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_indian_news_sentiment_en_1.0.0_3.0_1668058786154.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_indian_news_sentiment_en_1.0.0_3.0_1668058786154.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
      
embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
        .setInputCols(["document", "token"]) \
        .setOutputCol("embeddings")

sembeddings = nlp.SentenceEmbeddings()\
    .setInputCols(["document", "embeddings"]) \
    .setOutputCol("sentence_embeddings") \
    .setPoolingStrategy("AVERAGE")

classsifierdl = finance.ClassifierDLModel.pretrained("finclf_indian_news_sentiment", "en", "finance/models")\
                .setInputCols(["sentence_embeddings"])\
                .setOutputCol("label")

bert_clf_pipeline = Pipeline(stages=[document_assembler,
                                     tokenizer,
                                     embeddings,
                                     sembeddings,
                                     classsifierdl])

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
|Model Name:|finclf_indian_news_sentiment|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[label]|
|Language:|en|
|Size:|23.6 MB|

## References

An in-house augmented version of [this dataset](https://www.kaggle.com/datasets/harshrkh/india-financial-news-headlines-sentiments)

## Benchmarking

```bash

              precision    recall  f1-score   support

    NEGATIVE       0.75      0.78      0.76     21441
    POSITIVE       0.73      0.69      0.71     18449

    accuracy                           0.74     39890
   macro avg       0.74      0.74      0.74     39890
weighted avg       0.74      0.74      0.74     39890

```
