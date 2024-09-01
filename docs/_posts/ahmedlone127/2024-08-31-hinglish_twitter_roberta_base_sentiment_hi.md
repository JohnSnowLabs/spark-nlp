---
layout: model
title: Hindi hinglish_twitter_roberta_base_sentiment RoBertaForSequenceClassification from pascalrai
author: John Snow Labs
name: hinglish_twitter_roberta_base_sentiment
date: 2024-08-31
tags: [hi, open_source, onnx, sequence_classification, roberta]
task: Text Classification
language: hi
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hinglish_twitter_roberta_base_sentiment` is a Hindi model originally trained by pascalrai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hinglish_twitter_roberta_base_sentiment_hi_5.4.2_3.0_1725119297803.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hinglish_twitter_roberta_base_sentiment_hi_5.4.2_3.0_1725119297803.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier  = RoBertaForSequenceClassification.pretrained("hinglish_twitter_roberta_base_sentiment","hi") \
     .setInputCols(["documents","token"]) \
     .setOutputCol("class")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, sequenceClassifier])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")
    
val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier = RoBertaForSequenceClassification.pretrained("hinglish_twitter_roberta_base_sentiment", "hi")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("class") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hinglish_twitter_roberta_base_sentiment|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|hi|
|Size:|468.3 MB|

## References

https://huggingface.co/pascalrai/hinglish-twitter-roberta-base-sentiment