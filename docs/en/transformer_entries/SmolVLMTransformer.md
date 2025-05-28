{%- capture title -%}
SmolVLMTransformer
{%- endcapture -%}

{%- capture description -%}
Compact Multimodal Model for Visual Question Answering using SmolVLM.

SmolVLMTransformer can load SmolVLM models for visual question answering. The model consists of a vision encoder, a text encoder, and a text decoder. The vision encoder encodes the input image, the text encoder processes the input question alongside the image encoding, and the text decoder generates the answer to the question.

SmolVLM is a compact open multimodal model that accepts arbitrary sequences of image and text inputs to produce text outputs. Designed for efficiency, SmolVLM can answer questions about images, describe visual content, create stories grounded on multiple images, or function as a pure language model without visual inputs.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val visualQA = SmolVLMTransformer.pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("answer")
```
{%- endcapture -%}

{%- capture input_anno -%}
IMAGE
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
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

{%- endcapture -%}

{%- capture scala_example -%}
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
{%- endcapture -%}

{%- capture api_link -%}
[SmolVLMTransformer](/api/com/johnsnowlabs/nlp/annotators/cv/SmolVLMTransformer)
{%- endcapture -%}

{%- capture python_api_link -%}
[SmolVLMTransformer](/api/python/reference/autosummary/sparknlp/annotator/cv/smolvlm_transformer/index.html#sparknlp.annotator.cv.smolvlm_transformer.SmolVLMTransformer)
{%- endcapture -%}

{%- capture source_link -%}
[SmolVLMTransformer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/SmolVLMTransformer.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}