{%- capture title -%}
MLLamaForMultimodal
{%- endcapture -%}

{%- capture description -%}
Visual Question Answering using MLLama.

MLLamaForMultimodal can load LLAMA 3.2 Vision models for visual question answering.
The model consists of a vision encoder, a text encoder, and a text decoder.
The vision encoder encodes the input image, the text encoder processes the input question
alongside the image encoding, and the text decoder generates the answer to the question.

The Llama 3.2-Vision collection comprises pretrained and instruction-tuned multimodal large
language models (LLMs) available in 11B and 90B sizes. These models are optimized for visual
recognition, image reasoning, captioning, and answering general questions about images.
The models outperform many open-source and proprietary multimodal models on standard industry
benchmarks.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val visualQAClassifier = MLLamaForMultimodal.pretrained()
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
    lit("<|begin_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>What is unusual on this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
)
imageAssembler = ImageAssembler() \\
    .setInputCol("image") \\
    .setOutputCol("image_assembler")
visualQAClassifier = MLLamaForMultimodal.pretrained() \\
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

val testDF: DataFrame = imageDF.withColumn("text", lit("<|begin_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>What is unusual on this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"))

val imageAssembler: ImageAssembler = new ImageAssembler()
   .setInputCol("image")
   .setOutputCol("image_assembler")

val visualQAClassifier = MLLamaForMultimodal.pretrained()
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
[MLLamaForMultimodal](/api/com/johnsnowlabs/nlp/annotators/cv/MLLamaForMultimodal)
{%- endcapture -%}

{%- capture python_api_link -%}
[MLLamaForMultimodal](/api/python/reference/autosummary/sparknlp/annotator/cv/m_llama_for_multimodal/index.html#sparknlp.annotator.cv.mllama_for_multimodal.MLLamaForMultimodal)
{%- endcapture -%}

{%- capture source_link -%}
[MLLamaForMultimodal](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/MLLamaForMultimodal.scala)
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