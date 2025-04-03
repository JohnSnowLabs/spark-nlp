---
layout: model
title: English dog_food_vit_base_patch16_224_in21k ViTForImageClassification from sasha
author: John Snow Labs
name: dog_food_vit_base_patch16_224_in21k
date: 2025-04-03
tags: [en, open_source, onnx, image_classification, vit]
task: Image Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: ViTForImageClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained ViTForImageClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dog_food_vit_base_patch16_224_in21k` is a English model originally trained by sasha.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dog_food_vit_base_patch16_224_in21k_en_5.5.1_3.0_1743672462356.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dog_food_vit_base_patch16_224_in21k_en_5.5.1_3.0_1743672462356.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

		
		

image_assembler = ImageAssembler()\
  .setInputCol("image")\
  .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification.pretrained(""dog_food_vit_base_patch16_224_in21k","en")\
  .setInputCols("image_assembler")\
  .setOutputCol("class")

pipeline = Pipeline(stages=[
  image_assembler,
  imageClassifier,
])

pipelineModel = pipeline.fit(imageDF)

pipelineDF = pipelineModel.transform(imageDF)

```
```scala

val imageAssembler = new ImageAssembler()
    .setInputCol("image")
    .setOutputCol("image_assembler")

val imageClassifier =  ViTForImageClassification.pretrained("dog_food_vit_base_patch16_224_in21k","en") 
    .setInputCols("image_assembler") 
    .setOutputCol("class") 

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

val pipelineModel = pipeline.fit(imageDF)

val pipelineDF = pipelineModel.transform(imageDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dog_food_vit_base_patch16_224_in21k|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[label]|
|Language:|en|
|Size:|321.3 MB|

## References

https://huggingface.co/sasha/dog-food-vit-base-patch16-224-in21k