---
layout: model
title: twitter-xlm-roberta-base-sentiment
author: veerdhwaj
name: twitter_xlm_roberta_base_sentiment
date: 2023-07-28
tags: [sentiment, roberta, en, open_source, tensorflow]
task: Text Classification
language: en
edition: Spark NLP 5.0.0
spark_version: 3.2
supported: false
engine: tensorflow
annotator: XlmRoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a multilingual XLM-roBERTa-base model trained on ~198M tweets and finetuned for sentiment analysis. The sentiment fine-tuning was done on 8 languages (Ar, En, Fr, De, Hi, It, Sp, Pt) but it can be used for more languages
Huggingface : https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment

## Predicted Entities

`sentiment`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/veerdhwaj/twitter_xlm_roberta_base_sentiment_en_5.0.0_3.2_1690535217423.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://community.johnsnowlabs.com/veerdhwaj/twitter_xlm_roberta_base_sentiment_en_5.0.0_3.2_1690535217423.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val sequenceClassifier = XlmRoBertaForSequenceClassification.pretrained('twitter_xlm_roberta_base_sentiment')
  .setInputCols("token", "document")
  .setOutputCol("class")
  .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  sequenceClassifier
))

val data = Seq("I loved this movie when I was a child.", "It was pretty boring.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("class.result").show(false)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|twitter_xlm_roberta_base_sentiment|
|Compatibility:|Spark NLP 5.0.0+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.0 GB|
|Case sensitive:|true|
|Max sentence length:|512|