---
layout: model
title: Qwen2-VL
author: John Snow Labs
name: qwen2_vl_2b_instruct_int4
date: 2025-02-14
tags: [en, open_source, openvino]
task: Image Captioning
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: Qwen2VLTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

SoTA understanding of images of various resolution & ratio: Qwen2-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.

Understanding videos of 20min+: Qwen2-VL can understand videos over 20 minutes for high-quality video-based question answering, dialog, content creation, etc.

Agent that can operate your mobiles, robots, etc.: with the abilities of complex reasoning and decision making, Qwen2-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.

Multilingual Support: to serve global users, besides English and Chinese, Qwen2-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/qwen2_vl_2b_instruct_int4_en_5.5.1_3.0_1739495409030.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/qwen2_vl_2b_instruct_int4_en_5.5.1_3.0_1739495409030.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
test_df = image_df.withColumn("text", lit("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"))

imageAssembler = ImageAssembler()   
    .setInputCol("image")   
    .setOutputCol("image_assembler")

visualQAClassifier = Qwen2VLTransformer.pretrained()   
    .setInputCols("image_assembler")   
    .setOutputCol("answer")

pipeline = Pipeline().setStages([
    imageAssembler,
    visualQAClassifier
])

result = pipeline.fit(test_df).transform(test_df)
result.select("image_assembler.origin", "answer.result").show(false)
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.lit

val imageDF: DataFrame = spark.read
  .format("image")
  .option("dropInvalid", value = true)
  .load(imageFolder) // Replace with your image folder

val testDF: DataFrame = imageDF.withColumn("text", lit("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"))

val imageAssembler: ImageAssembler = new ImageAssembler()
   .setInputCol("image")
   .setOutputCol("image_assembler")

val visualQAClassifier = Qwen2VLTransformer.pretrained()
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
|Model Name:|qwen2_vl_2b_instruct_int4|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|1.5 GB|

## References

https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct