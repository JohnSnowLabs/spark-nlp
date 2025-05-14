---
layout: model
title: Gemma 3 4B PT Int4
author: John Snow Labs
name: gemma_3_4b_pt_int4
date: 2025-04-28
tags: [en, open_source, openvino]
task: Image Captioning
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: Gemma3ForMultiModal
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Gemma3ForMultiModal can load Gemma 3 Vision models for visual question answering.
The model consists of a vision encoder, a text encoder, a text decoder and a model merger.
The vision encoder will encode the input image, the text encoder will encode the input text,
the model merger will merge the image and text embeddings, and the text decoder will output the answer.

Gemma 3 is a family of lightweight, state-of-the-art open models from Google, built from the same 
research and technology used to create the Gemini models. It features:
- Large 128K context window
- Multilingual support in over 140 languages
- Multimodal capabilities handling both text and image inputs
- Optimized for deployment on limited resources (laptops, desktops, cloud)

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gemma_3_4b_pt_int4_en_5.5.1_3.0_1745822550666.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gemma_3_4b_pt_int4_en_5.5.1_3.0_1745822550666.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
test_df = image_df.withColumn("text", lit("<bos><start_of_turn>user\nYou are a helpful assistant.\n\n<start_of_image>Describe this image in detail.<end_of_turn>\n<start_of_turn>model\n"))

imageAssembler = ImageAssembler()   
          .setInputCol("image")   
          .setOutputCol("image_assembler")

visualQAClassifier = Gemma3ForMultiModal.pretrained()   
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

val testDF: DataFrame = imageDF.withColumn("text", lit("<bos><start_of_turn>user\nYou are a helpful assistant.\n\n<start_of_image>Describe this image in detail.<end_of_turn>\n<start_of_turn>model\n"))

val imageAssembler: ImageAssembler = new ImageAssembler()
     .setInputCol("image")
     .setOutputCol("image_assembler")

val visualQAClassifier = Gemma3ForMultiModal.pretrained()
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
|Model Name:|gemma_3_4b_pt_int4|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|3.1 GB|