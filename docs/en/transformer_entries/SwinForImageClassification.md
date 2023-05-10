{%- capture title -%}
SwinForImageClassification
{%- endcapture -%}

{%- capture description -%}
SwinImageClassification is an image classifier based on Swin.

The Swin Transformer was proposed in Swin Transformer: Hierarchical Vision Transformer using
Shifted Windows by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin,
Baining Guo.

It is basically a hierarchical Transformer whose representation is computed with shifted
windows. The shifted windowing scheme brings greater efficiency by limiting self-attention
computation to non-overlapping local windows while also allowing for cross-window connection.

Pretrained models can be loaded with `pretrained` of the companion object:

```
val imageClassifier = SwinForImageClassification.pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("class")
```

The default model is `"image_classifier_swin_base_patch_4_window_7_224"`, if no name is
provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Image+Classification).

Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
examples, see
[SwinForImageClassificationTest](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/SwinForImageClassificationTest.scala).

**References:**

[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)

**Paper Abstract:**

*This paper presents a new vision Transformer, called Swin Transformer, that capably serves
as a general-purpose backbone for computer vision. Challenges in adapting Transformer from
language to vision arise from differences between the two domains, such as large variations in
the scale of visual entities and the high resolution of pixels in images compared to words in
text. To address these differences, we propose a hierarchical Transformer whose representation
is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by
limiting self-attention computation to non-overlapping local windows while also allowing for
cross-window connection. This hierarchical architecture has the flexibility to model at
various scales and has linear computational complexity with respect to image size. These
qualities of Swin Transformer make it compatible with a broad range of vision tasks, including
image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as
object detection (58.7 box AP and 51.1 mask AP on COCO test- dev) and semantic segmentation
(53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the- art by a large
margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the
potential of Transformer-based models as vision backbones. The hierarchical design and the
shifted window approach also prove beneficial for all-MLP architectures.*
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

imageDF: DataFrame = spark.read \
    .format("image") \
    .option("dropInvalid", value = True) \
    .load("src/test/resources/image/")

imageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = SwinForImageClassification \
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

val imageClassifier = SwinForImageClassification
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
[SwinForImageClassification](/api/com/johnsnowlabs/nlp/annotators/cv/SwinForImageClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[SwinForImageClassification](/api/python/reference/autosummary/sparknlp/annotator/cv/swin_for_image_classification/index.html#sparknlp.annotator.cv.swin_for_image_classification.SwinForImageClassification)
{%- endcapture -%}

{%- capture source_link -%}
[SwinForImageClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/SwinForImageClassification.scala)
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