---
layout: model
title: English facial_emotions_image_detection_rafdb_microsoft_vit SwinForImageClassification from adhityamw11
author: John Snow Labs
name: facial_emotions_image_detection_rafdb_microsoft_vit
date: 2025-02-02
tags: [en, open_source, onnx, image_classification, swin]
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

Pretrained SwinForImageClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`facial_emotions_image_detection_rafdb_microsoft_vit` is a English model originally trained by adhityamw11.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/facial_emotions_image_detection_rafdb_microsoft_vit_en_5.5.1_3.0_1738507809423.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/facial_emotions_image_detection_rafdb_microsoft_vit_en_5.5.1_3.0_1738507809423.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
image_assembler = ImageAssembler()\
  .setInputCol("image")\
  .setOutputCol("image_assembler")

imageClassifier = SwinForImageClassification.pretrained(""facial_emotions_image_detection_rafdb_microsoft_vit","en")\
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

val imageClassifier =  SwinForImageClassification.pretrained("facial_emotions_image_detection_rafdb_microsoft_vit","en") 
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
|Model Name:|facial_emotions_image_detection_rafdb_microsoft_vit|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[label]|
|Language:|en|
|Size:|103.2 MB|

## References

References

https://huggingface.co/adhityamw11/facial_emotions_image_detection_rafdb_microsoft_vit