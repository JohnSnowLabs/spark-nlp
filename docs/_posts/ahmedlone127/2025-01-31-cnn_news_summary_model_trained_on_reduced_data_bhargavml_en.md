---
layout: model
title: English cnn_news_summary_model_trained_on_reduced_data_bhargavml T5Transformer from bhargavml
author: John Snow Labs
name: cnn_news_summary_model_trained_on_reduced_data_bhargavml
date: 2025-01-31
tags: [en, open_source, onnx, t5, question_answering, summarization, translation, text_generation]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cnn_news_summary_model_trained_on_reduced_data_bhargavml` is a English model originally trained by bhargavml.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cnn_news_summary_model_trained_on_reduced_data_bhargavml_en_5.5.1_3.0_1738303609856.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cnn_news_summary_model_trained_on_reduced_data_bhargavml_en_5.5.1_3.0_1738303609856.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

t5  = T5Transformer.pretrained("cnn_news_summary_model_trained_on_reduced_data_bhargavml","en") \
     .setInputCols(["document"]) \
     .setOutputCol("output")

pipeline = Pipeline().setStages([documentAssembler, t5])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")

val t5 = T5Transformer.pretrained("cnn_news_summary_model_trained_on_reduced_data_bhargavml", "en")
    .setInputCols(Array("documents")) 
    .setOutputCol("output") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cnn_news_summary_model_trained_on_reduced_data_bhargavml|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|en|
|Size:|269.4 MB|

## References

https://huggingface.co/bhargavml/cnn_news_summary_model_trained_on_reduced_data