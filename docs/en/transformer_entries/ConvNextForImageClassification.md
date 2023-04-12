{%- capture title -%}
ConvNextForImageClassification
{%- endcapture -%}

{%- capture description -%}
ConvNextForImageClassification is an image classifier based on ConvNet models.

The ConvNeXT model was proposed in A ConvNet for the 2020s by Zhuang Liu, Hanzi Mao, Chao-Yuan
Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie. ConvNeXT is a pure convolutional
model (ConvNet), inspired by the design of Vision Transformers, that claims to outperform
them.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val imageClassifier = ConvNextForImageClassification.pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("class")
```

The default model is `"image_classifier_convnext_tiny_224_local"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Image+Classification).

Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
examples, see
[ConvNextForImageClassificationTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/ConvNextForImageClassificationTestSpec.scala).

**References:**

[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

**Paper Abstract:**

*The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers
(ViTs), which quickly superseded ConvNets as the state-of-the-art image classification model.
A vanilla ViT, on the other hand, faces difficulties when applied to general computer vision
tasks such as object detection and semantic segmentation. It is the hierarchical Transformers
(e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers
practically viable as a generic vision backbone and demonstrating remarkable performance on a
wide variety of vision tasks. However, the effectiveness of such hybrid approaches is still
largely credited to the intrinsic superiority of Transformers, rather than the inherent
inductive biases of convolutions. In this work, we reexamine the design spaces and test the
limits of what a pure ConvNet can achieve. We gradually "modernize" a standard ResNet toward
the design of a vision Transformer, and discover several key components that contribute to the
performance difference along the way. The outcome of this exploration is a family of pure
ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts
compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8%
ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K
segmentation, while maintaining the simplicity and efficiency of standard ConvNets.*
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
imageDF = spark.read \
    .format("image") \
    .option("dropInvalid", value = True) \
    .load("src/test/resources/image/")
imageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")
imageClassifier = ConvNextForImageClassification \
    .pretrained() \
    .setInputCols(["image_assembler"]) \
    .setOutputCol("class")
pipeline = Pipeline().setStages([imageAssembler, imageClassifier])
pipelineDF = pipeline.fit(imageDF).transform(imageDF)
pipelineDF \
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "class.result") \
  .show(truncate=False)
+-----------------+----------------------------------------------------------+
|image_name       |result                                                    |
+-----------------+----------------------------------------------------------+
|bluetick.jpg     |[bluetick]                                                |
|chihuahua.jpg    |[Chihuahua]                                               |
|egyptian_cat.jpeg|[tabby, tabby cat]                                        |
|hen.JPEG         |[hen]                                                     |
|hippopotamus.JPEG|[hippopotamus, hippo, river horse, Hippopotamus amphibius]|
|junco.JPEG       |[junco, snowbird]                                         |
|ostrich.JPEG     |[ostrich, Struthio camelus]                               |
|ox.JPEG          |[ox]                                                      |
|palace.JPEG      |[palace]                                                  |
|tractor.JPEG     |[thresher, thrasher, threshing machine                    |
+-----------------+----------------------------------------------------------+
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

val imageClassifier = ConvNextForImageClassification
  .pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))
val pipelineDF = pipeline.fit(imageDF).transform(imageDF)

pipelineDF
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "class.result")
  .show(truncate = false)
+-----------------+----------------------------------------------------------+
|image_name       |result                                                    |
+-----------------+----------------------------------------------------------+
|palace.JPEG      |[palace]                                                  |
|egyptian_cat.jpeg|[tabby, tabby cat]                                        |
|hippopotamus.JPEG|[hippopotamus, hippo, river horse, Hippopotamus amphibius]|
|hen.JPEG         |[hen]                                                     |
|ostrich.JPEG     |[ostrich, Struthio camelus]                               |
|junco.JPEG       |[junco, snowbird]                                         |
|bluetick.jpg     |[bluetick]                                                |
|chihuahua.jpg    |[Chihuahua]                                               |
|tractor.JPEG     |[tractor]                                                 |
|ox.JPEG          |[ox]                                                      |
+-----------------+----------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[ConvNextForImageClassification](/api/com/johnsnowlabs/nlp/annotators/cv/ConvNextForImageClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[ConvNextForImageClassification](/api/python/reference/autosummary/sparknlp/annotator/cv/convnext_for_image_classification/index.html#sparknlp.annotator.cv.convnext_for_image_classification.ConvNextForImageClassification)
{%- endcapture -%}

{%- capture source_link -%}
[ConvNextForImageClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/ConvNextForImageClassification.scala)
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