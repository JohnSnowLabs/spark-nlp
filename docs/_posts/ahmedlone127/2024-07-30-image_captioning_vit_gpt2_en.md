---
layout: model
title: Image Caption with VisionEncoderDecoder ViT GPT2
author: John Snow Labs
name: image_captioning_vit_gpt2
date: 2024-07-30
tags: [en, image_classification, vit, gpt2, captioning, open_source, tensorflow]
task: Image Captioning
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: tensorflow
annotator: VisionEncoderDecoderForImageCaptioning
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is an image captioning model using ViT to encode images and GPT2 to generate captions. Original model from https://huggingface.co/nlpconnect/vit-gpt2-image-captioning

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_captioning_vit_gpt2_en_5.4.2_3.0_1722335452448.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/image_captioning_vit_gpt2_en_5.4.2_3.0_1722335452448.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
imageDF = spark.read \
    .format("image") \
    .option("dropInvalid", value = True) \
    .load("src/test/resources/image/")
imageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")
imageCaptioning = VisionEncoderDecoderForImageCaptioning \
    .pretrained() \
    .setBeamSize(2) \
    .setDoSample(False) \
    .setInputCols(["image_assembler"]) \
    .setOutputCol("caption")
pipeline = Pipeline().setStages([imageAssembler, imageCaptioning])
pipelineDF = pipeline.fit(imageDF).transform(imageDF)
pipelineDF \
    .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "caption.result")     .show(truncate = False)

```
```scala

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.ImageAssembler
import org.apache.spark.ml.Pipeline

val imageDF: DataFrame = spark.read
  .format("image")
  .option("dropInvalid", value = true)
  .load("src/test/resources/image/")

val imageCaptioning = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val imageClassifier = VisionEncoderDecoderForImageCaptioning
  .pretrained()
  .setBeamSize(2)
  .setDoSample(false)
  .setInputCols("image_assembler")
  .setOutputCol("caption")

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageCaptioning))
val pipelineDF = pipeline.fit(imageDF).transform(imageDF)

pipelineDF
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "caption.result")
  .show(truncate = false)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|image_captioning_vit_gpt2|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[caption]|
|Language:|en|
|Size:|897.1 MB|