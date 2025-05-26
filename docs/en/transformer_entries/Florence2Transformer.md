{%- capture title -%}
Florence2Transformer
{%- endcapture -%}

{%- capture description -%}
Florence2Transformer can load Florence-2 models for a wide variety of vision and vision-language tasks using prompt-based inference.

Florence-2 is an advanced vision foundation model from Microsoft that uses a prompt-based approach to handle tasks like image captioning, object detection, segmentation, OCR, and more. The model leverages the FLD-5B dataset, containing 5.4 billion annotations across 126 million images, to master multi-task learning. Its sequence-to-sequence architecture enables it to excel in both zero-shot and fine-tuned settings.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val florence2 = Florence2Transformer.pretrained()
     .setInputCols("image_assembler")
     .setOutputCol("answer")
```
The default model is `"florence2_base_ft_int4"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?task=Vision+Tasks).

==Supported Tasks==

Florence-2 supports a variety of tasks through prompt engineering. The following prompt tokens can be used:

- <CAPTION>: Image captioning
- <DETAILED_CAPTION>: Detailed image captioning
- <MORE_DETAILED_CAPTION>: Paragraph-level captioning
- <CAPTION_TO_PHRASE_GROUNDING>: Phrase grounding from caption (requires additional text input)
- <OD>: Object detection
- <DENSE_REGION_CAPTION>: Dense region captioning
- <REGION_PROPOSAL>: Region proposal
- <OCR>: Optical Character Recognition (plain text extraction)
- <OCR_WITH_REGION>: OCR with region information
- <REFERRING_EXPRESSION_SEGMENTATION>: Segmentation for a referred phrase (requires additional text input)
- <REGION_TO_SEGMENTATION>: Polygon mask for a region (requires additional text input)
- <OPEN_VOCABULARY_DETECTION>: Open vocabulary detection for a phrase (requires additional text input)
- <REGION_TO_CATEGORY>: Category of a region (requires additional text input)
- <REGION_TO_DESCRIPTION>: Description of a region (requires additional text input)
- <REGION_TO_OCR>: OCR for a region (requires additional text input)

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
test_df = image_df.withColumn("text", lit("<OD>"))

imageAssembler = ImageAssembler()   
          .setInputCol("image")   
          .setOutputCol("image_assembler")

florence2 = Florence2Transformer.pretrained()   
          .setInputCols(["image_assembler"])   
          .setOutputCol("answer")

pipeline = Pipeline().setStages([
          imageAssembler,
          florence2
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

val testDF: DataFrame = imageDF.withColumn("text", lit("<OD>"))

val imageAssembler: ImageAssembler = new ImageAssembler()
     .setInputCol("image")
     .setOutputCol("image_assembler")

val florence2 = Florence2Transformer.pretrained()
     .setInputCols("image_assembler")
     .setOutputCol("answer")

val pipeline = new Pipeline().setStages(Array(
     imageAssembler,
     florence2
))

val result = pipeline.fit(testDF).transform(testDF)

result.select("image_assembler.origin", "answer.result").show(false)
{%- endcapture -%}

{%- capture api_link -%}
[Florence2Transformer](/api/com/johnsnowlabs/nlp/annotators/cv/Florence2Transformer)
{%- endcapture -%}

{%- capture python_api_link -%}
[Florence2Transformer](/api/python/reference/autosummary/sparknlp/annotator/cv/florence2_transformer/index.html#sparknlp.annotator.cv.florence2_transformer.Florence2Transformer)
{%- endcapture -%}

{%- capture source_link -%}
[Florence2Transformer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/Florence2Transformer.scala)
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