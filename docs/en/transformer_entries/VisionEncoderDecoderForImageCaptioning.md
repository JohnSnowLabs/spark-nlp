{%- capture title -%}
VisionEncoderDecoderForImageCaptioning
{%- endcapture -%}

{%- capture description -%}
VisionEncoderDecoder model that converts images into text captions. It allows for the use of
pretrained vision auto-encoding models, such as ViT, BEiT, or DeiT as the encoder, in
combination with pretrained language models, like RoBERTa, GPT2, or BERT as the decoder.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val imageClassifier = VisionEncoderDecoderForImageCaptioning.pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("caption")
```

The default model is `"image_captioning_vit_gpt2"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Image+Captioning).

Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
examples, see
[VisionEncoderDecoderTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/VisionEncoderDecoderTestSpec.scala).

*Note:*

This is a very computationally expensive module especially on larger batch sizes. The use of an
accelerator such as GPU is recommended.
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
imageDF = spark.read \
    .format("image") \
    .option("dropInvalid", value = True) \
    .load("src/test/resources/image/")
imageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")
imageCaptioning = VisionEncoderDecoderForImageCaptioning \
    .pretrained() \
    .setBeamSize(2) \
    .setDoSample(False) \
    .setInputCols(["image_assembler"]) \
    .setOutputCol("caption")
pipeline = Pipeline().setStages([imageAssembler, imageCaptioning])
pipelineDF = pipeline.fit(imageDF).transform(imageDF)
pipelineDF \
    .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "caption.result") \
    .show(truncate = False)
+-----------------+---------------------------------------------------------+
|image_name       |result                                                   |
+-----------------+---------------------------------------------------------+
|palace.JPEG      |[a large room filled with furniture and a large window]  |
|egyptian_cat.jpeg|[a cat laying on a couch next to another cat]            |
|hippopotamus.JPEG|[a brown bear in a body of water]                        |
|hen.JPEG         |[a flock of chickens standing next to each other]        |
|ostrich.JPEG     |[a large bird standing on top of a lush green field]     |
|junco.JPEG       |[a small bird standing on a wet ground]                  |
|bluetick.jpg     |[a small dog standing on a wooden floor]                 |
|chihuahua.jpg    |[a small brown dog wearing a blue sweater]               |
|tractor.JPEG     |[a man is standing in a field with a tractor]            |
|ox.JPEG          |[a large brown cow standing on top of a lush green field]|
+-----------------+---------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.ImageAssembler
import org.apache.spark.ml.Pipeline

val imageDF: DataFrame = spark.read
  .format("image")
  .option("dropInvalid", value = true)
  .load("src/test/resources/image/")

val imageCaptioning = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val imageClassifier = VisionEncoderDecoderForImageCaptioning
  .pretrained()
  .setBeamSize(2)
  .setDoSample(false)
  .setInputCols("image_assembler")
  .setOutputCol("caption")

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageCaptioning))
val pipelineDF = pipeline.fit(imageDF).transform(imageDF)

pipelineDF
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "caption.result")
  .show(truncate = false)

+-----------------+---------------------------------------------------------+
|image_name       |result                                                   |
+-----------------+---------------------------------------------------------+
|palace.JPEG      |[a large room filled with furniture and a large window]  |
|egyptian_cat.jpeg|[a cat laying on a couch next to another cat]            |
|hippopotamus.JPEG|[a brown bear in a body of water]                        |
|hen.JPEG         |[a flock of chickens standing next to each other]        |
|ostrich.JPEG     |[a large bird standing on top of a lush green field]     |
|junco.JPEG       |[a small bird standing on a wet ground]                  |
|bluetick.jpg     |[a small dog standing on a wooden floor]                 |
|chihuahua.jpg    |[a small brown dog wearing a blue sweater]               |
|tractor.JPEG     |[a man is standing in a field with a tractor]            |
|ox.JPEG          |[a large brown cow standing on top of a lush green field]|
+-----------------+---------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[VisionEncoderDecoderForImageCaptioning](/api/com/johnsnowlabs/nlp/annotators/cv/VisionEncoderDecoderForImageCaptioning)
{%- endcapture -%}

{%- capture python_api_link -%}
[VisionEncoderDecoderForImageCaptioning](/api/python/reference/autosummary/sparknlp/annotator/cv/vision_encoder_decoder_for_image_captioning/index.html#sparknlp.annotator.cv.vision_encoder_decoder_for_image_captioning.VisionEncoderDecoderForImageCaptioning)
{%- endcapture -%}

{%- capture source_link -%}
[VisionEncoderDecoderForImageCaptioning](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/VisionEncoderDecoderForCaptioning.scala)
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