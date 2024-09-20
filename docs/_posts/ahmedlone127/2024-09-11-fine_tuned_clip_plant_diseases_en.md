---
layout: model
title: English fine_tuned_clip_plant_diseases CLIPForZeroShotClassification from muhammad-atif-ali
author: John Snow Labs
name: fine_tuned_clip_plant_diseases
date: 2024-09-11
tags: [en, open_source, onnx, zero_shot, clip, image]
task: Zero-Shot Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: CLIPForZeroShotClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CLIPForZeroShotClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fine_tuned_clip_plant_diseases` is a English model originally trained by muhammad-atif-ali.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fine_tuned_clip_plant_diseases_en_5.5.0_3.0_1726057760495.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fine_tuned_clip_plant_diseases_en_5.5.0_3.0_1726057760495.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

imageClassifier = CLIPForZeroShotClassification.pretrained("fine_tuned_clip_plant_diseases","en") \
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
  
val imageClassifier = CLIPForZeroShotClassification.pretrained("fine_tuned_clip_plant_diseases","en") \
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
|Model Name:|fine_tuned_clip_plant_diseases|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[label]|
|Language:|en|
|Size:|567.3 MB|

## References

https://huggingface.co/muhammad-atif-ali/fine-tuned-clip-plant-diseases