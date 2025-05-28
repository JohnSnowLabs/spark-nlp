---
layout: model
title: InternVL 3 2B int4
author: John Snow Labs
name: internvl3_2b_int4
date: 2025-05-16
tags: [en, openvino, open_source]
task: Image Captioning
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: InternVLForMultiModal
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Visual Question Answering using InternVL.

InternVLForMultiModal can load InternVL Vision models for visual question answering.
The model consists of a vision encoder, a text encoder, a text decoder and a model merger.
The vision encoder will encode the input image, the text encoder will encode the input text,
the model merger will merge the image and text embeddings, and the text decoder will output the answer.

InternVL 2.5 is an advanced multimodal large language model (MLLM) series that builds upon InternVL 2.0,
maintaining its core model architecture while introducing significant enhancements in training and testing
strategies as well as data quality. Key features include:
- Large context window support
- Multilingual support
- Multimodal capabilities handling both text and image inputs
- Optimized for deployment with int4 quantization

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/internvl3_2b_int4_en_5.5.1_3.0_1747371321772.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/internvl3_2b_int4_en_5.5.1_3.0_1747371321772.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
test_df = image_df.withColumn("text", lit("<|im_start|><image>\nDescribe this image in detail.<|im_end|><|im_start|>assistant\n"))

imageAssembler = ImageAssembler()   
          .setInputCol("image")   
          .setOutputCol("image_assembler")

visualQAClassifier = InternVLForMultiModal.pretrained()   
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

val testDF: DataFrame = imageDF.withColumn("text", lit("<|im_start|><image>\nDescribe this image in detail.<|im_end|><|im_start|>assistant\n"))

val imageAssembler: ImageAssembler = new ImageAssembler()
     .setInputCol("image")
     .setOutputCol("image_assembler")

val visualQAClassifier = InternVLForMultiModal.pretrained()
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
|Model Name:|internvl3_2b_int4|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|1.3 GB|