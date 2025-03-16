{%- capture title -%}
AutoGGUFVisionModel
{%- endcapture -%}

{%- capture description -%}
Multimodal annotator that uses the llama.cpp library to generate text completions with large
language models. It supports ingesting images for captioning.

At the moment only CLIP based models are supported.

For settable parameters, and their explanations, see HasLlamaCppInferenceProperties,
HasLlamaCppModelProperties and refer to the llama.cpp documentation of
[server.cpp](https://github.com/ggerganov/llama.cpp/tree/7d5e8777ae1d21af99d4f95be10db4870720da91/examples/server)
for more information.

If the parameters are not set, the annotator will default to use the parameters provided by
the model.

This annotator expects a column of annotator type AnnotationImage for the image and
Annotation for the caption. Note that the image bytes in the image annotation need to be
raw image bytes without preprocessing. We provide the helper function
ImageAssembler.loadImagesAsBytes to load the image bytes from a directory.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val autoGGUFVisionModel = AutoGGUFVisionModel.pretrained()
  .setInputCols("image", "document")
  .setOutputCol("completions")
```

The default model is `"llava_v1.5_7b_Q4_0_gguf"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models).

For extended examples of usage, see the
[AutoGGUFVisionModelTest](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFVisionModelTest.scala)
and the
[example notebook](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama.cpp_in_Spark_NLP_AutoGGUFVisionModel.ipynb).

**Note**: To use GPU inference with this annotator, make sure to use the Spark NLP GPU package and set
the number of GPU layers with the `setNGpuLayers` method.

When using larger models, we recommend adjusting GPU usage with `setNCtx` and `setNGpuLayers`
according to your hardware to avoid out-of-memory errors.
{%- endcapture -%}

{%- capture input_anno -%}
IMAGE, DOCUMENT
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

documentAssembler = DocumentAssembler() \
    .setInputCol("caption") \
    .setOutputCol("caption_document")
imageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imagesPath = "src/test/resources/image/"
data = ImageAssembler \
    .loadImagesAsBytes(spark, imagesPath) \
    .withColumn("caption", lit("Caption this image.")) # Add a caption to each image.

nPredict = 40
model = AutoGGUFVisionModel.pretrained() \
    .setInputCols(["caption_document", "image_assembler"]) \
    .setOutputCol("completions") \
    .setBatchSize(4) \
    .setNGpuLayers(99) \
    .setNCtx(4096) \
    .setMinKeep(0) \
    .setMinP(0.05) \
    .setNPredict(nPredict) \
    .setNProbs(0) \
    .setPenalizeNl(False) \
    .setRepeatLastN(256) \
    .setRepeatPenalty(1.18) \
    .setStopStrings(["</s>", "Llama:", "User:"]) \
    .setTemperature(0.05) \
    .setTfsZ(1) \
    .setTypicalP(1) \
    .setTopK(40) \
    .setTopP(0.95)

pipeline = Pipeline().setStages([documentAssembler, imageAssembler, model])
pipeline.fit(data).transform(data) \
    .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "completions.result") \
    .show(truncate = False)
+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|image_name       |result                                                                                                                                                                                        |
+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|palace.JPEG      |[ The image depicts a large, ornate room with high ceilings and beautifully decorated walls. There are several chairs placed throughout the space, some of which have cushions]               |
|egyptian_cat.jpeg|[ The image features two cats lying on a pink surface, possibly a bed or sofa. One cat is positioned towards the left side of the scene and appears to be sleeping while holding]             |
|hippopotamus.JPEG|[ A large brown hippo is swimming in a body of water, possibly an aquarium. The hippo appears to be enjoying its time in the water and seems relaxed as it floats]                            |
|hen.JPEG         |[ The image features a large chicken standing next to several baby chickens. In total, there are five birds in the scene: one adult and four young ones. They appear to be gathered together] |
|ostrich.JPEG     |[ The image features a large, long-necked bird standing in the grass. It appears to be an ostrich or similar species with its head held high and looking around. In addition to]              |
|junco.JPEG       |[ A small bird with a black head and white chest is standing on the snow. It appears to be looking at something, possibly food or another animal in its vicinity. The scene takes place out]  |
|bluetick.jpg     |[ A dog with a red collar is sitting on the floor, looking at something. The dog appears to be staring into the distance or focusing its attention on an object in front of it.]              |
|chihuahua.jpg    |[ A small brown dog wearing a sweater is sitting on the floor. The dog appears to be looking at something, possibly its owner or another animal in the room. It seems comfortable and relaxed]|
|tractor.JPEG     |[ A man is sitting in the driver's seat of a green tractor, which has yellow wheels and tires. The tractor appears to be parked on top of an empty field with]                                |
|ox.JPEG          |[ A large bull with horns is standing in a grassy field.]                                                                                                                                     |
+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.ImageAssembler
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.lit

val documentAssembler = new DocumentAssembler()
  .setInputCol("caption")
  .setOutputCol("caption_document")

val imageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val imagesPath = "src/test/resources/image/"
val data: DataFrame = ImageAssembler
  .loadImagesAsBytes(ResourceHelper.spark, imagesPath)
  .withColumn("caption", lit("Caption this image.")) // Add a caption to each image.

val nPredict = 40
val model = AutoGGUFVisionModel.pretrained()
  .setInputCols("caption_document", "image_assembler")
  .setOutputCol("completions")
  .setBatchSize(4)
  .setNGpuLayers(99)
  .setNCtx(4096)
  .setMinKeep(0)
  .setMinP(0.05f)
  .setNPredict(nPredict)
  .setNProbs(0)
  .setPenalizeNl(false)
  .setRepeatLastN(256)
  .setRepeatPenalty(1.18f)
  .setStopStrings(Array("</s>", "Llama:", "User:"))
  .setTemperature(0.05f)
  .setTfsZ(1)
  .setTypicalP(1)
  .setTopK(40)
  .setTopP(0.95f)

val pipeline = new Pipeline().setStages(Array(documentAssembler, imageAssembler, model))
pipeline
  .fit(data)
  .transform(data)
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "completions.result")
  .show(truncate = false)
+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|image_name       |result                                                                                                                                                                                        |
+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|palace.JPEG      |[ The image depicts a large, ornate room with high ceilings and beautifully decorated walls. There are several chairs placed throughout the space, some of which have cushions]               |
|egyptian_cat.jpeg|[ The image features two cats lying on a pink surface, possibly a bed or sofa. One cat is positioned towards the left side of the scene and appears to be sleeping while holding]             |
|hippopotamus.JPEG|[ A large brown hippo is swimming in a body of water, possibly an aquarium. The hippo appears to be enjoying its time in the water and seems relaxed as it floats]                            |
|hen.JPEG         |[ The image features a large chicken standing next to several baby chickens. In total, there are five birds in the scene: one adult and four young ones. They appear to be gathered together] |
|ostrich.JPEG     |[ The image features a large, long-necked bird standing in the grass. It appears to be an ostrich or similar species with its head held high and looking around. In addition to]              |
|junco.JPEG       |[ A small bird with a black head and white chest is standing on the snow. It appears to be looking at something, possibly food or another animal in its vicinity. The scene takes place out]  |
|bluetick.jpg     |[ A dog with a red collar is sitting on the floor, looking at something. The dog appears to be staring into the distance or focusing its attention on an object in front of it.]              |
|chihuahua.jpg    |[ A small brown dog wearing a sweater is sitting on the floor. The dog appears to be looking at something, possibly its owner or another animal in the room. It seems comfortable and relaxed]|
|tractor.JPEG     |[ A man is sitting in the driver's seat of a green tractor, which has yellow wheels and tires. The tractor appears to be parked on top of an empty field with]                                |
|ox.JPEG          |[ A large bull with horns is standing in a grassy field.]                                                                                                                                     |
+-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture api_link -%}
[AutoGGUFVisionModel](/api/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFVisionModel)
{%- endcapture -%}

{%- capture python_api_link -%}
[AutoGGUFVisionModel](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/auto_gguf_vision_model/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[AutoGGUFVisionModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFVisionModel.scala)
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