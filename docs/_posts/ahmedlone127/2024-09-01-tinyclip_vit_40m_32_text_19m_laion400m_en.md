---
layout: model
title: English tinyclip_vit_40m_32_text_19m_laion400m CLIPForZeroShotClassification from wkcn
author: John Snow Labs
name: tinyclip_vit_40m_32_text_19m_laion400m
date: 2024-09-01
tags: [en, open_source, onnx, zero_shot, clip, image]
task: Zero-Shot Classification
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: CLIPForZeroShotClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CLIPForZeroShotClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tinyclip_vit_40m_32_text_19m_laion400m` is a English model originally trained by wkcn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tinyclip_vit_40m_32_text_19m_laion400m_en_5.4.2_3.0_1725159209286.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tinyclip_vit_40m_32_text_19m_laion400m_en_5.4.2_3.0_1725159209286.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

imageClassifier = CLIPForZeroShotClassification.pretrained("tinyclip_vit_40m_32_text_19m_laion400m","en") \
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
  
val imageClassifier = CLIPForZeroShotClassification.pretrained("tinyclip_vit_40m_32_text_19m_laion400m","en") \
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
|Model Name:|tinyclip_vit_40m_32_text_19m_laion400m|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[label]|
|Language:|en|
|Size:|202.5 MB|

## References

https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M