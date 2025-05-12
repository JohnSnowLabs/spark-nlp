---
layout: model
title: English octfusion_exp1_hkdb_balanced ViTForImageClassification from g30rv17ys
author: John Snow Labs
name: octfusion_exp1_hkdb_balanced
date: 2025-04-05
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

Pretrained ViTForImageClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`octfusion_exp1_hkdb_balanced` is a English model originally trained by g30rv17ys.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/octfusion_exp1_hkdb_balanced_en_5.5.1_3.0_1743831759619.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/octfusion_exp1_hkdb_balanced_en_5.5.1_3.0_1743831759619.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

		
		

image_assembler = ImageAssembler()\
  .setInputCol("image")\
  .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification.pretrained(""octfusion_exp1_hkdb_balanced","en")\
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

val imageClassifier =  ViTForImageClassification.pretrained("octfusion_exp1_hkdb_balanced","en") 
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
|Model Name:|octfusion_exp1_hkdb_balanced|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[label]|
|Language:|en|
|Size:|321.3 MB|

## References

https://huggingface.co/g30rv17ys/OCTFusion-Exp1-HKDB-Balanced