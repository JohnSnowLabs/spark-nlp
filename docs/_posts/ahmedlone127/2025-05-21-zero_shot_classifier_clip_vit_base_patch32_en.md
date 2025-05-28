---
layout: model
title: Image Zero Shot Classification with CLIP
author: John Snow Labs
name: zero_shot_classifier_clip_vit_base_patch32
date: 2025-05-21
tags: [classification, image, en, zero_shot, open_source, onnx, openvino]
task: Zero-Shot Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: CLIPForZeroShotClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

CLIP (Contrastive Language-Image Pre-Training) is a neural network that was trained on image
and text pairs. It has the ability to predict images without training on any hard-coded
labels. This makes it very flexible, as labels can be provided during inference. This is
similar to the zero-shot capabilities of the GPT-2 and 3 models.

This model was imported from huggingface transformers:
https://huggingface.co/openai/clip-vit-base-patch32

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/zero_shot_classifier_clip_vit_base_patch32_en_5.5.1_3.0_1747820900697.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/zero_shot_classifier_clip_vit_base_patch32_en_5.5.1_3.0_1747820900697.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

imageAssembler: ImageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

candidateLabels = [
    "a photo of a bird",
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a hen",
    "a photo of a hippo",
    "a photo of a room",
    "a photo of a tractor",
    "a photo of an ostrich",
    "a photo of an ox"]

imageClassifier = CLIPForZeroShotClassification \
    .pretrained() \
    .setInputCols(["image_assembler"]) \
    .setOutputCol("label") \
    .setCandidateLabels(candidateLabels)

pipeline = Pipeline().setStages([imageAssembler, imageClassifier])
pipelineDF = pipeline.fit(imageDF).transform(imageDF)
pipelineDF \
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "label.result") \
  .show(truncate=False)
```
```scala
import com.johnsnowlabs.nlp.ImageAssembler
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
val imageDF = ResourceHelper.spark.read
  .format("image")
  .option("dropInvalid", value = true)
  .load("src/test/resources/image/")
val imageAssembler: ImageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")
val candidateLabels = Array(
  "a photo of a bird",
  "a photo of a cat",
  "a photo of a dog",
  "a photo of a hen",
  "a photo of a hippo",
  "a photo of a room",
  "a photo of a tractor",
  "a photo of an ostrich",
  "a photo of an ox")
val imageClassifier = CLIPForZeroShotClassification
  .pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("label")
  .setCandidateLabels(candidateLabels)
val pipeline =
  new Pipeline().setStages(Array(imageAssembler, imageClassifier)).fit(imageDF).transform(imageDF)
pipeline
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "label.result")
  .show(truncate = false)
```
</div>

## Results

```bash


+-----------------+-----------------------+
|image_name       |result                 |
+-----------------+-----------------------+
|palace.JPEG      |[a photo of a room]    |
|egyptian_cat.jpeg|[a photo of a cat]     |
|hippopotamus.JPEG|[a photo of a hippo]   |
|hen.JPEG         |[a photo of a hen]     |
|ostrich.JPEG     |[a photo of an ostrich]|
|junco.JPEG       |[a photo of a bird]    |
|bluetick.jpg     |[a photo of a dog]     |
|chihuahua.jpg    |[a photo of a dog]     |
|tractor.JPEG     |[a photo of a tractor] |
|ox.JPEG          |[a photo of an ox]     |
+-----------------+-----------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|zero_shot_classifier_clip_vit_base_patch32|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[label]|
|Language:|en|
|Size:|397.5 MB|

## References

References

https://huggingface.co/openai/clip-vit-base-patch32