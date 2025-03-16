---
layout: model
title: SwinForImageClassification - image_classifier_swin_tiny_patch4_window7_224
author: John Snow Labs
name: image_classifier_swin_tiny_patch4_window7_224
date: 2025-01-24
tags: [open_source, swin, image, en, english, image_classification, imagenet, onnx]
task: Image Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: SwinForImageClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Swin model for Image Classification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.

Swin Transformer was introduced in the paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) by Liu et al.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_swin_tiny_patch4_window7_224_en_5.5.1_3.0_1737715181288.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/image_classifier_swin_tiny_patch4_window7_224_en_5.5.1_3.0_1737715181288.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
image_assembler = ImageAssembler()\
  .setInputCol("image")\
  .setOutputCol("image_assembler")

imageClassifier = SwinForImageClassification.pretrained("image_classifier_swin_tiny_patch4_window7_224", "en")\
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

val imageClassifier = SwinForImageClassification
    .pretrained("image_classifier_swin_tiny_patch4_window7_224", "en")
    .setInputCols("image_assembler") 
    .setOutputCol("class") 

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

val pipelineModel = pipeline.fit(imageDF)

val pipelineDF = pipelineModel.transform(imageDF)
```

{:.nlu-block}
```python
import nlu
import requests
response = requests.get('https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp/master/docs/assets/images/hen.JPEG')
with open('hen.JPEG', 'wb') as f:
    f.write(response.content)
nlu.load("en.classify_image.swin.tiny").predict("hen.JPEG")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|image_classifier_swin_tiny_patch4_window7_224|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[label]|
|Language:|en|
|Size:|212.1 MB|

## References

References

[https://huggingface.co/microsoft/swin-tiny-patch4-window7-224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)