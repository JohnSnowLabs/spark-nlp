{%- capture title -%}
Qwen2VLTransformer
{%- endcapture -%}

{%- capture description -%}
Visual Question Answering and Multimodal Instruction Following using Qwen2-VL.

Qwen2VLTransformer can load Qwen2 Vision-Language models for visual question answering and
multimodal instruction following. The model consists of a vision encoder, a text encoder, and
a text decoder. The vision encoder processes the input image, the text encoder integrates
the encoding of the image with the input text, and the text decoder outputs the response to
the query or instruction.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val visualQA = Qwen2VLTransformer.pretrained()
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
{%- endcapture -%}

{%- capture api_link -%}
[Qwen2VLTransformer](/api/com/johnsnowlabs/nlp/annotators/cv/Qwen2VLTransformer)
{%- endcapture -%}

{%- capture python_api_link -%}
[Qwen2VLTransformer](/api/python/reference/autosummary/sparknlp/annotator/cv/qwen2_vl/index.html#sparknlp.annotator.cv.qwen2_vl.Qwen2VLTransformer)
{%- endcapture -%}

{%- capture source_link -%}
[Qwen2VLTransformer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/Qwen2VLTransformer.scala)
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