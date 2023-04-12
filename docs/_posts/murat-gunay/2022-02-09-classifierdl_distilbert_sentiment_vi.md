---
layout: model
title: Sentiment Analysis of Vietnamese texts
author: John Snow Labs
name: classifierdl_distilbert_sentiment
date: 2022-02-09
tags: [vietnamese, sentiment, vi, open_source]
task: Text Classification
language: vi
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: ClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model identifies Positive or Negative sentiments in Vietnamese texts.

## Predicted Entities

`POSITIVE`, `NEGATIVE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_distilbert_sentiment_vi_3.4.0_3.0_1644408533716.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_distilbert_sentiment_vi_3.4.0_3.0_1644408533716.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\
    .setCleanupMode("shrink")

tokenizer = Tokenizer() \
      .setInputCols(["document"]) \
      .setOutputCol("token")
    
normalizer = Normalizer() \
      .setInputCols(["token"]) \
      .setOutputCol("normalized")

lemmatizer = LemmatizerModel.pretrained("lemma", "vi") \
        .setInputCols(["normalized"]) \
        .setOutputCol("lemma")

distilbert = DistilBertEmbeddings.pretrained("distilbert_base_cased", "vi")\
  .setInputCols(["document",'token'])\
  .setOutputCol("embeddings")\
  .setCaseSensitive(False)

embeddingsSentence = SentenceEmbeddings() \
      .setInputCols(["document", "embeddings"]) \
      .setOutputCol("sentence_embeddings") \
      .setPoolingStrategy("AVERAGE")

sentimentClassifier = ClassifierDLModel.pretrained('classifierdl_distilbert_sentiment', 'vi') \
  .setInputCols(["document", "sentence_embeddings"]) \
  .setOutputCol("class")

vi_sentiment_pipeline = Pipeline(stages=[document, tokenizer, normalizer, lemmatizer, distilbert, embeddingsSentence, sentimentClassifier])

light_pipeline = LightPipeline(vi_sentiment_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
result = light_pipeline.annotate("Chất cotton siêu đẹp mịn mát.")
result["class"]
```
```scala
val document = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    .setCleanupMode("shrink")

val tokenizer = Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")
    
val normalizer = Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")

val lemmatizer = LemmatizerModel.pretrained("lemma", "vi")
        .setInputCols(Array("normalized"))
        .setOutputCol("lemma")

val distilbert = DistilBertEmbeddings.pretrained("distilbert_base_cased", "vi")
  .setInputCols(Array("document","token"))
  .setOutputCol("embeddings")
  .setCaseSensitive(False)

val embeddingsSentence = SentenceEmbeddings()
      .setInputCols(Array("document", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

val sentimentClassifier = ClassifierDLModel.pretrained.("classifierdl_distilbert_sentiment", "vi")
  .setInputCols(Array("document", "sentence_embeddings"))
  .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document, tokenizer, normalizer, lemmatizer, distilbert, embeddingsSentence, sentimentClassifier))

val light_pipeline = LightPipeline(pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

val result = light_pipeline.annotate("Chất cotton siêu đẹp mịn mát.")
```
</div>

## Results

```bash
['POSITIVE']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_distilbert_sentiment|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|vi|
|Size:|23.6 MB|

## References

[https://www.kaggle.com/datvuthanh/vietnamese-sentiment](https://www.kaggle.com/datvuthanh/vietnamese-sentiment)

## Benchmarking

```bash
       label  precision    recall  f1-score   support
    NEGATIVE       0.88      0.79      0.83       956
    POSITIVE       0.80      0.89      0.84       931
    accuracy          -         -      0.84      1887
   macro-avg       0.84      0.84      0.84      1887
weighted-avg       0.84      0.84      0.84      1887
```
