---
layout: model
title: English image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_marsch TFSwinForImageClassification from marsch
author: John Snow Labs
name: image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_marsch
date: 2023-03-23
tags: [swin, en, image, open_source, image_classification, imagenet, tensorflow]
task: Image Classification
language: en
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: SwinForImageClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Swin  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_marsch` is a English model originally trained by marsch.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_marsch_en_4.4.0_3.0_1679595961309.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_marsch_en_4.4.0_3.0_1679595961309.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler  = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier  = SwinForImageClassification \
    .pretrained("image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_marsch", "en") \
    .setInputCols("image_assembler") \
    .setOutputCol("class")

pipeline = Pipeline(stages=[
  image_assembler,
  imageClassifier,
])

pipelineModel = pipeline.fit(imageDF)

pipelineDF = pipelineModel.transform(imageDF)
```
```scala

val image_assembler = new ImageAssembler() 
    .setInputCol("image") 
    .setOutputCol("image_assembler")

val imageClassifier = SwinForImageClassification
    .pretrained("image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_marsch", "en")
    .setInputCols("image_assembler") 
    .setOutputCol("class") 

val pipeline = new Pipeline().setStages(Array(image_assembler, imageClassifier))

val pipelineModel = pipeline.fit(imageDF)

val pipelineDF = pipelineModel.transform(imageDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_marsch|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|210.5 MB|