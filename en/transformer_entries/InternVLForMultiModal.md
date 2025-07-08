{%- capture title -%}
InternVLForMultiModal
{%- endcapture -%}

{%- capture description -%}
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

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val visualQA = InternVLForMultiModal.pretrained()
     .setInputCols("image_assembler")
     .setOutputCol("answer")
```
The default model is `"internvl2_5_1b_int4"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Question+Answering).

To see which models are compatible and how to import them see
[Import Transformers into Spark NLP 🚀](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669).

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
{%- endcapture -%}

{%- capture scala_example -%}
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
{%- endcapture -%}

{%- capture api_link -%}
[InternVLForMultiModal](/api/com/johnsnowlabs/nlp/annotators/cv/InternVLForMultiModal)
{%- endcapture -%}

{%- capture python_api_link -%}
[InternVLForMultiModal](/api/python/reference/autosummary/sparknlp/annotator/cv/internvl_for_multimodal/index.html#sparknlp.annotator.cv.internvl_for_multimodal.InternVLForMultiModal)
{%- endcapture -%}

{%- capture source_link -%}
[InternVLForMultiModal](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/InternVLForMultiModal.scala)
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