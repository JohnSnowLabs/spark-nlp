---
layout: model
title: Telugu telugu_dataset_other_sentiment_distilbert DistilBertForSequenceClassification from Sathvik6323
author: John Snow Labs
name: telugu_dataset_other_sentiment_distilbert
date: 2024-01-02
tags: [bert, te, open_source, sequence_classification, onnx]
task: Text Classification
language: te
edition: Spark NLP 5.2.2
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`telugu_dataset_other_sentiment_distilbert` is a Telugu model originally trained by Sathvik6323.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/telugu_dataset_other_sentiment_distilbert_te_5.2.2_3.0_1704175935108.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/telugu_dataset_other_sentiment_distilbert_te_5.2.2_3.0_1704175935108.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
sequenceClassifier = DistilBertForSequenceClassification.pretrained("telugu_dataset_other_sentiment_distilbert","te")\
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
    
val sequenceClassifier = DistilBertForSequenceClassification.pretrained("telugu_dataset_other_sentiment_distilbert","te")
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
|Model Name:|telugu_dataset_other_sentiment_distilbert|
|Compatibility:|Spark NLP 5.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[class]|
|Language:|te|
|Size:|249.4 MB|

## References

https://huggingface.co/Sathvik6323/Telugu_dataset_other_sentiment_distilbert