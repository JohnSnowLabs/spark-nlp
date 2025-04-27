{%- capture title -%}
Gemma3ForMultiModal
{%- endcapture -%}

{%- capture description -%}
Visual Question Answering using Gemma 3 Vision.

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

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val visualQA = Gemma3ForMultiModal.pretrained()
     .setInputCols("image_assembler")
     .setOutputCol("answer")
```
The default model is `"gemma3_4b_it_int4"`, if no name is provided.

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
{%- endcapture -%}

{%- capture api_link -%}
[Gemma3ForMultiModal](https://www.google.com/url?sa=E&source=gmail&q=/api/com/johnsnowlabs/nlp/annotators/cv/Gemma3ForMultiModal)
{%- endcapture -%}

{%- capture python_api_link -%}
[Gemma3ForMultiModal](https://www.google.com/url?sa=E&source=gmail&q=/api/python/reference/autosummary/sparknlp/annotator/cv/gemma3_for_multimodal/index.html#sparknlp.annotator.cv.gemma3_for_multimodal.Gemma3ForMultiModal)
{%- endcapture -%}

{%- capture source_link -%}
[Gemma3ForMultiModal](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/Gemma3ForMultiModal.scala)
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