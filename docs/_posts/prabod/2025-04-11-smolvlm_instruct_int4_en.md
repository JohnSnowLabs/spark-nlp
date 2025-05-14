---
layout: model
title: SmolVLM by HUggingface
author: John Snow Labs
name: smolvlm_instruct_int4
date: 2025-04-11
tags: [en, openvino, vlm, open_source]
task: Image Captioning
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: SmolVLMTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

SmolVLM is a compact open multimodal model that accepts arbitrary sequences of image and text inputs to produce text outputs. Designed for efficiency, SmolVLM can answer questions about images, describe visual content, create stories grounded on multiple images, or function as a pure language model without visual inputs. Its lightweight architecture makes it suitable for on-device applications while maintaining strong performance on multimodal tasks.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/smolvlm_instruct_int4_en_5.5.1_3.0_1744355673028.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/smolvlm_instruct_int4_en_5.5.1_3.0_1744355673028.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
test_df = image_df.withColumn(
    "text",
    lit("<|im_start|>User:<image>Can you describe the image?<end_of_utterance>\nAssistant:")
)
imageAssembler = ImageAssembler() \\
    .setInputCol("image") \\
    .setOutputCol("image_assembler")
visualQAClassifier = SmolVLMTransformer.pretrained() \\
    .setInputCols("image_assembler") \\
    .setOutputCol("answer")
pipeline = Pipeline().setStages([
    imageAssembler,
    visualQAClassifier
])
result = pipeline.fit(test_df).transform(test_df)
result.select("image_assembler.origin", "answer.result").show(truncate=False)
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

val testDF: DataFrame = imageDF.withColumn("text", lit("<|im_start|>User:<image>Can you describe the image?<end_of_utterance>\nAssistant:"))

val imageAssembler: ImageAssembler = new ImageAssembler()
   .setInputCol("image")
   .setOutputCol("image_assembler")

val visualQAClassifier = SmolVLMTransformer.pretrained()
   .setInputCols("image_assembler")
   .setOutputCol("answer")

val pipeline = new Pipeline().setStages(Array(
  imageAssembler,
  visualQAClassifier
))

val result = pipeline.fit(testDF).transform(testDF)

result.select("image_assembler.origin", "answer.result").show(truncate=false)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|smolvlm_instruct_int4|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|1.8 GB|

## References

https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct