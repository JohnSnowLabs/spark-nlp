{%- capture title -%}
ViTForImageClassification
{%- endcapture -%}

{%- capture description -%}
Vision Transformer (ViT) for image classification.

ViT is a transformer based alternative to the convolutional neural networks usually used for
image recognition tasks.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val imageClassifier = ViTForImageClassification.pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("class")
```
The default model is `"image_classifier_vit_base_patch16_224"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Image+Classification).

Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. The
Spark NLP Workshop example shows how to import them
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
examples, see
[ViTImageClassificationTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/ViTImageClassificationTestSpec.scala).

**References:**

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

**Paper Abstract:**

*While the Transformer architecture has become the de-facto standard for natural language
processing tasks, its applications to computer vision remain limited. In vision, attention is
either applied in conjunction with convolutional networks, or used to replace certain
components of convolutional networks while keeping their overall structure in place. We show
that this reliance on CNNs is not necessary and a pure transformer applied directly to
sequences of image patches can perform very well on image classification tasks. When
pre-trained on large amounts of data and transferred to multiple mid-sized or small image
recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains
excellent results compared to state-of-the-art convolutional networks while requiring
substantially fewer computational resources to train.*
{%- endcapture -%}

{%- capture input_anno -%}
IMAGE
{%- endcapture -%}

{%- capture output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

imageDF: DataFrame = spark.read \\
    .format("image") \\
    .option("dropInvalid", value = True) \\
    .load("src/test/resources/image/")
imageAssembler = ImageAssembler() \\
    .setInputCol("image") \\
    .setOutputCol("image_assembler")
imageClassifier = ViTForImageClassification \\
    .pretrained() \\
    .setInputCols(["image_assembler"]) \\
    .setOutputCol("class")

pipeline = Pipeline().setStages([imageAssembler, imageClassifier])
pipelineDF = pipeline.fit(imageDF).transform(imageDF)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.ImageAssembler
import org.apache.spark.ml.Pipeline

val imageDF: DataFrame = spark.read
  .format("image")
  .option("dropInvalid", value = true)
  .load("src/test/resources/image/")

val imageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val imageClassifier = ViTForImageClassification
  .pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))
val pipelineDF = pipeline.fit(imageDF).transform(imageDF)

{%- endcapture -%}

{%- capture api_link -%}
[ViTForImageClassification](/api/com/johnsnowlabs/nlp/annotators/cv/ViTForImageClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[ViTForImageClassification](/api/python/reference/autosummary/sparknlp/annotator/cv/vit_for_image_classification/index.html#sparknlp.annotator.cv.vit_for_image_classification.ViTForImageClassification)
{%- endcapture -%}

{%- capture source_link -%}
[ViTForImageClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/ViTForImageClassification.scala)
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