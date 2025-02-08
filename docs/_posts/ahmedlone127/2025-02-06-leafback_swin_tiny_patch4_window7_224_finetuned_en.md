---
layout: model
title: English leafback_swin_tiny_patch4_window7_224_finetuned SwinForImageClassification from b07611031
author: John Snow Labs
name: leafback_swin_tiny_patch4_window7_224_finetuned
date: 2025-02-06
tags: [en, open_source, onnx, image_classification, swin]
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

Pretrained SwinForImageClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`leafback_swin_tiny_patch4_window7_224_finetuned` is a English model originally trained by b07611031.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/leafback_swin_tiny_patch4_window7_224_finetuned_en_5.5.1_3.0_1738830988489.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/leafback_swin_tiny_patch4_window7_224_finetuned_en_5.5.1_3.0_1738830988489.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

		
		

image_assembler = ImageAssembler()\
  .setInputCol("image")\
  .setOutputCol("image_assembler")

imageClassifier = SwinForImageClassification.pretrained(""leafback_swin_tiny_patch4_window7_224_finetuned","en")\
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

val imageClassifier =  SwinForImageClassification.pretrained("leafback_swin_tiny_patch4_window7_224_finetuned","en") 
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
|Model Name:|leafback_swin_tiny_patch4_window7_224_finetuned|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[label]|
|Language:|en|
|Size:|206.3 MB|

## References

https://huggingface.co/b07611031/leafback-swin-tiny-patch4-window7-224-finetuned