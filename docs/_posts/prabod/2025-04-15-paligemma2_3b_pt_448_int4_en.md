---
layout: model
title: PaliGemma2 3B pt 448 int4
author: John Snow Labs
name: paligemma2_3b_pt_448_int4
date: 2025-04-15
tags: [en, open_source, openvino]
task: Image Captioning
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: PaliGemmaForMultiModal
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

PaliGemma 2 is an update of the PaliGemma vision-language model (VLM) which incorporates the capabilities of the Gemma 2 models. The PaliGemma family of models is inspired by PaLI-3 and based on open components such as the SigLIP vision model and Gemma 2 language models. It takes both image and text as input and generates text as output, supporting multiple languages. It is designed for class-leading fine-tune performance on a wide range of vision-language tasks such as image and short video caption, visual question answering, text reading, object detection and object segmentation.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/paligemma2_3b_pt_448_int4_en_5.5.1_3.0_1744693861187.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/paligemma2_3b_pt_448_int4_en_5.5.1_3.0_1744693861187.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import lit

image_df = spark.read.format("image").load(path=images_path) # Replace with your image path
test_df = image_df.withColumn("text", lit("USER: \n <image> \nDescribe this image. \nASSISTANT:\n"))

imageAssembler = ImageAssembler()   
          .setInputCol("image")   
          .setOutputCol("image_assembler")

visualQAClassifier = PaliGemmaForMultiModal.pretrained()   
          .setInputCols("image_assembler")   
          .setOutputCol("answer")

pipeline = Pipeline().setStages([
          imageAssembler,
          visualQAClassifier
])

result = pipeline.fit(test_df).transform(test_df)
result.select("image_assembler.origin", "answer.result").show(False)
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.lit

val imageFolder = "path/to/your/images" // Replace with your image path

val imageDF: DataFrame = spark.read
     .format("image")
     .option("dropInvalid", value = true)
     .load(imageFolder)

val testDF: DataFrame = imageDF.withColumn("text", lit("USER: \n <image> \nDescribe this image. \nASSISTANT:\n"))

val imageAssembler: ImageAssembler = new ImageAssembler()
     .setInputCol("image")
     .setOutputCol("image_assembler")

val visualQAClassifier = PaliGemmaForMultiModal.pretrained()
     .setInputCols("image_assembler")
     .setOutputCol("answer")

val pipeline = new Pipeline().setStages(Array(
     imageAssembler,
     visualQAClassifier
))

val result = pipeline.fit(testDF).transform(testDF)

result.select("image_assembler.origin", "answer.result").show(false)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|paligemma2_3b_pt_448_int4|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|3.0 GB|

## References

https://huggingface.co/google/paligemma2-3b-mix-224