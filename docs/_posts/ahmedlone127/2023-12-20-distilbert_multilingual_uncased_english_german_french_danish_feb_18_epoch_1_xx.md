---
layout: model
title: Multilingual distilbert_multilingual_uncased_english_german_french_danish_feb_18_epoch_1 DistilBertForSequenceClassification from somm
author: John Snow Labs
name: distilbert_multilingual_uncased_english_german_french_danish_feb_18_epoch_1
date: 2023-12-20
tags: [bert, xx, open_source, sequence_classification, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.2.1
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_multilingual_uncased_english_german_french_danish_feb_18_epoch_1` is a Multilingual model originally trained by somm.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_multilingual_uncased_english_german_french_danish_feb_18_epoch_1_xx_5.2.1_3.0_1703030766192.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_multilingual_uncased_english_german_french_danish_feb_18_epoch_1_xx_5.2.1_3.0_1703030766192.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")  
    
sequenceClassifier = DistilBertForSequenceClassification.pretrained("distilbert_multilingual_uncased_english_german_french_danish_feb_18_epoch_1","xx")\
            .setInputCols(["document","token"])\
            .setOutputCol("class")

pipeline = Pipeline().setStages([document_assembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)

```
```scala

val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document") 
    .setOutputCol("token")  
    
val sequenceClassifier = DistilBertForSequenceClassification.pretrained("distilbert_multilingual_uncased_english_german_french_danish_feb_18_epoch_1","xx")
            .setInputCols(Array("document","token"))
            .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_multilingual_uncased_english_german_french_danish_feb_18_epoch_1|
|Compatibility:|Spark NLP 5.2.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[class]|
|Language:|xx|
|Size:|507.6 MB|

## References

https://huggingface.co/somm/distilbert-multilingual-uncased-en-de-fr-da-feb-18-epoch-1