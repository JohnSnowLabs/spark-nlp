---
layout: model
title: English RoBertaForSequenceClassification Cased model (from cardiffnlp)
author: John Snow Labs
name: roberta_classifier_tweet_topic_19_multi
date: 2022-09-19
tags: [en, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `tweet-topic-19-multi` is a English model originally trained by `cardiffnlp`.

## Predicted Entities

`film_tv_&_video`, `diaries_&_daily_life`, `other_hobbies`, `music`, `business_&_entrepreneurs`, `relationships`, `gaming`, `youth_&_student_life`, `food_&_dining`, `fitness_&_health`, `news_&_social_concern`, `fashion_&_style`, `family`, `travel_&_adventure`, `science_&_technology`, `learning_&_educational`, `arts_&_culture`, `sports`, `celebrity_&_pop_culture`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_tweet_topic_19_multi_en_4.1.0_3.0_1663618637987.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_classifier_tweet_topic_19_multi_en_4.1.0_3.0_1663618637987.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_tweet_topic_19_multi","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_tweet_topic_19_multi","en") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_tweet_topic_19_multi|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|469.0 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/cardiffnlp/tweet-topic-19-multi
- https://github.com/cardiffnlp/tweeteval
- https://arxiv.org/abs/2202.03829
- https://github.com/cardiffnlp/timelms