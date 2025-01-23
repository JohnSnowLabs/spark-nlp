---
layout: model
title: English clip_vit_b_32_laion2b_s34b_b79k CLIPForZeroShotClassification from timsek
author: John Snow Labs
name: clip_vit_b_32_laion2b_s34b_b79k
date: 2025-01-23
tags: [en, open_source, onnx, zero_shot, clip, image]
task: Zero-Shot Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: CLIPForZeroShotClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CLIPForZeroShotClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`clip_vit_b_32_laion2b_s34b_b79k` is a English model originally trained by timsek.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clip_vit_b_32_laion2b_s34b_b79k_en_5.5.1_3.0_1737629417079.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clip_vit_b_32_laion2b_s34b_b79k_en_5.5.1_3.0_1737629417079.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

imageDF = spark.read \
    .format("image") \
    .option("dropInvalid", value = True) \
    .load("src/test/resources/image/")
    
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

ImageAssembler = ImageAssembler() \
	.setInputCol("image") \
	.setOutputCol("image_assembler")

imageClassifier = CLIPForZeroShotClassification.pretrained("clip_vit_b_32_laion2b_s34b_b79k","en") \
     .setInputCols(["image_assembler"]) \
     .setOutputCol("label") \
     .setCandidateLabels(candidateLabels)

pipeline = Pipeline().setStages([ImageAssembler, imageClassifier])
pipelineModel = pipeline.fit(imageDF)
pipelineDF = pipelineModel.transform(imageDF)


```
```scala

		
val imageDF = ResourceHelper.spark.read
  .format("image")
  .option("dropInvalid", value = true)
  .load("src/test/resources/image/")

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

val imageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")
  
val imageClassifier = CLIPForZeroShotClassification.pretrained("clip_vit_b_32_laion2b_s34b_b79k","en") \
     .setInputCols(Array("image_assembler")) \
     .setOutputCol("label") \
     .setCandidateLabels(candidateLabels)
  
val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))
val pipelineModel = pipeline.fit(imageDF)
val pipelineDF = pipelineModel.transform(imageDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clip_vit_b_32_laion2b_s34b_b79k|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[label]|
|Language:|en|
|Size:|567.6 MB|

## References

https://huggingface.co/timsek/CLIP-ViT-B-32-laion2B-s34B-b79K