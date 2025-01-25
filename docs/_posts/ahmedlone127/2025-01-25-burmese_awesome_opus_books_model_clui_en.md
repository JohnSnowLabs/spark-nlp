---
layout: model
title: English burmese_awesome_opus_books_model_clui MarianTransformer from clui
author: John Snow Labs
name: burmese_awesome_opus_books_model_clui
date: 2025-01-25
tags: [en, open_source, onnx, translation, marian]
task: Translation
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: MarianTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`burmese_awesome_opus_books_model_clui` is a English model originally trained by clui.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/burmese_awesome_opus_books_model_clui_en_5.5.1_3.0_1737782025757.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/burmese_awesome_opus_books_model_clui_en_5.5.1_3.0_1737782025757.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

sentenceDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") \
      .setInputCols(["document"]) \
      .setOutputCol("translation")

marian = MarianTransformer.pretrained("burmese_awesome_opus_books_model_clui","en") \
      .setInputCols(["sentence"]) \
      .setOutputCol("embeddings")       
        
pipeline = Pipeline().setStages([documentAssembler, sentenceDL, marian])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val marian = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val embeddings = MarianTransformer.pretrained("burmese_awesome_opus_books_model_clui","en") 
    .setInputCols(Array("sentence")) 
    .setOutputCol("translation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDL, marian))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|burmese_awesome_opus_books_model_clui|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentences]|
|Output Labels:|[translation]|
|Language:|en|
|Size:|477.1 MB|

## References

https://huggingface.co/clui/my_awesome_opus_books_model