{%- capture title -%}
JanusForMultiModal
{%- endcapture -%}

{%- capture description -%}
Unified Multimodal Understanding and Generation using Janus.

JanusForMultiModal can load Janus Vision models for unified multimodal understanding and generation.
The model consists of a vision encoder, a text encoder, and a text decoder. Janus decouples visual encoding for enhanced flexibility, leveraging a unified transformer architecture for both understanding and generation tasks.

Janus uses SigLIP-L as the vision encoder, supporting 384 x 384 image inputs. For image generation, it utilizes a tokenizer with a downsample rate of 16. The framework is based on DeepSeek-LLM-1.3b-base, trained on approximately 500B text tokens.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val visualQA = JanusForMultiModal.pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("answer")
```
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
    lit("User: <image_placeholder>Describe image in details\n\nAssistant:")
)
imageAssembler = ImageAssembler() \\
    .setInputCol("image") \\
    .setOutputCol("image_assembler")
visualQAClassifier = JanusForMultiModal.pretrained() \\
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

val testDF: DataFrame = imageDF.withColumn("text", lit("User: <image_placeholder>Describe image in details\n\nAssistant:"))

val imageAssembler: ImageAssembler = new ImageAssembler()
   .setInputCol("image")
   .setOutputCol("image_assembler")

val visualQAClassifier = JanusForMultiModal.pretrained()
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
[JanusForMultiModal](/api/com/johnsnowlabs/nlp/annotators/cv/JanusForMultiModal)
{%- endcapture -%}

{%- capture python_api_link -%}
[JanusForMultiModal](/api/python/reference/autosummary/sparknlp/annotator/cv/janus_for_multimodal/index.html#sparknlp.annotator.cv.janus_for_multimodal.JanusForMultiModal)
{%- endcapture -%}

{%- capture source_link -%}
[JanusForMultiModal](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/JanusForMultiModal.scala)
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
