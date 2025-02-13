{%- capture title -%}
Phi3Vision
{%- endcapture -%}

{%- capture description -%}
Visual Question Answering using Phi3Vision.

Phi3Vision can load Phi3Vision models for visual question answering.
The model consists of a vision encoder, a text encoder as well as a text decoder.
The vision encoder will encode the input image, the text encoder will encode the input question together
with the encoding of the image, and the text decoder will output the answer to the question.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val visualQA = Phi3Vision.pretrained()
     .setInputCols("image_assembler")
     .setOutputCol("answer")
```

The default model is `"phi_3_vision_128k_instruct"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Question+Answering).

Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
see which models are compatible and how to import them see
[Import Transformers into Spark NLP 🚀](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669).

For extended examples of usage, see
[Phi3VisionTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/Phi3VisionTest.scala).

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
test_df = image_df.withColumn("text", lit("<|user|> \n <|image_1|> \nWhat is unusual on this picture? <|end|>\n <|assistant|>\n"))

imageAssembler = ImageAssembler()   
          .setInputCol("image")   
          .setOutputCol("image_assembler")

visualQAClassifier = Phi3Vision.pretrained("phi_3_vision_128k_instruct","en")   
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

val testDF: DataFrame = imageDF.withColumn("text", lit("<|user|> \n <|image_1|> \nWhat is unusual on this picture? <|end|>\n <|assistant|>\n"))

val imageAssembler: ImageAssembler = new ImageAssembler()
     .setInputCol("image")
     .setOutputCol("image_assembler")

val visualQAClassifier = Phi3Vision.pretrained("phi_3_vision_128k_instruct","en")
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
[Phi3Vision](https://www.google.com/url?sa=E&source=gmail&q=/api/com/johnsnowlabs/nlp/annotators/cv/Phi3Vision)
{%- endcapture -%}

{%- capture python_api_link -%}
[Phi3Vision](https://www.google.com/url?sa=E&source=gmail&q=/api/python/reference/autosummary/sparknlp/annotator/cv/phi3_vision/index.html#sparknlp.annotator.cv.phi3_vision.Phi3Vision)
{%- endcapture -%}

{%- capture source_link -%}
[Phi3Vision](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/Phi3Vision.scala)
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