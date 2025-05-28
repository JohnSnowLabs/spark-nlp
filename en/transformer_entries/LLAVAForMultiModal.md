{%- capture title -%}
LLAVAForMultiModal
{%- endcapture -%}

{%- capture description -%}
Visual Question Answering using LLAVA.

LLAVAForMultiModal can load LLAVA models for visual question answering.
The model consists of a vision encoder, a text encoder as well as a text decoder.
The vision encoder will encode the input image, the text encoder will encode the input question together
with the encoding of the image, and the text decoder will output the answer to the question.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val visualQA = LLAVAForMultiModal.pretrained()
     .setInputCols("image_assembler")
     .setOutputCol("answer")
```
The default model is `"llava_1_5_7b_hf"`, if no name is provided.

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
test_df = image_df.withColumn("text", lit("USER: \n <|image|> \n What's this picture about? \n ASSISTANT:\n"))

imageAssembler = ImageAssembler()   
          .setInputCol("image")   
          .setOutputCol("image_assembler")

visualQAClassifier = LLAVAForMultiModal.pretrained()   
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

val testDF: DataFrame = imageDF.withColumn("text", lit("USER: \n <|image|> \nWhat is unusual on this picture? \n ASSISTANT:\n"))

val imageAssembler: ImageAssembler = new ImageAssembler()
     .setInputCol("image")
     .setOutputCol("image_assembler")

val visualQAClassifier = LLAVAForMultiModal.pretrained()
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
[LLAVAForMultiModal](/api/com/johnsnowlabs/nlp/annotators/cv/LLAVAForMultiModal)
{%- endcapture -%}

{%- capture python_api_link -%}
[LLAVAForMultiModal](/api/python/reference/autosummary/sparknlp/annotator/cv/llava_for_multimodal/index.html#sparknlp.annotator.cv.llava_for_multimodal.LLAVAForMultiModal)
{%- endcapture -%}

{%- capture source_link -%}
[LLAVAForMultiModal](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/LLAVAForMultiModal.scala)
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