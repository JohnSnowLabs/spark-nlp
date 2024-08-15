{%- capture title -%}
CLIPForZeroShotClassification
{%- endcapture -%}

{%- capture description -%}
Zero Shot Image Classifier based on CLIP.

CLIP (Contrastive Language-Image Pre-Training) is a neural network that was trained on image
and text pairs. It has the ability to predict images without training on any hard-coded
labels. This makes it very flexible, as labels can be provided during inference. This is
similar to the zero-shot capabilities of the GPT-2 and 3 models.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val imageClassifier = CLIPForZeroShotClassification.pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("label")
```

The default model is `"zero_shot_classifier_clip_vit_base_patch32"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Zero-Shot+Classification).

Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
see which models are compatible and how to import them see
[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669) and to see more extended
examples, see
[CLIPForZeroShotClassificationTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/CLIPForZeroShotClassificationTestSpec.scala).
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

imageAssembler: ImageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

candidateLabels = [
    "a photo of a bird",
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a hen",
    "a photo of a hippo",
    "a photo of a room",
    "a photo of a tractor",
    "a photo of an ostrich",
    "a photo of an ox"]

imageClassifier = CLIPForZeroShotClassification \
    .pretrained() \
    .setInputCols(["image_assembler"]) \
    .setOutputCol("label") \
    .setCandidateLabels(candidateLabels)

pipeline = Pipeline().setStages([imageAssembler, imageClassifier])
pipelineDF = pipeline.fit(imageDF).transform(imageDF)
pipelineDF \
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "label.result") \
  .show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.ImageAssembler
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val imageDF = ResourceHelper.spark.read
  .format("image")
  .option("dropInvalid", value = true)
  .load("src/test/resources/image/")

val imageAssembler: ImageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val candidateLabels = Array(
  "a photo of a bird",
  "a photo of a cat",
  "a photo of a dog",
  "a photo of a hen",
  "a photo of a hippo",
  "a photo of a room",
  "a photo of a tractor",
  "a photo of an ostrich",
  "a photo of an ox")

val model = CLIPForZeroShotClassification
  .pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("label")
  .setCandidateLabels(candidateLabels)

val pipeline =
  new Pipeline().setStages(Array(imageAssembler, model)).fit(imageDF).transform(imageDF)

pipeline
  .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "label.result")
  .show(truncate = false)
+-----------------+-----------------------+
|image_name       |result                 |
+-----------------+-----------------------+
|palace.JPEG      |[a photo of a room]    |
|egyptian_cat.jpeg|[a photo of a cat]     |
|hippopotamus.JPEG|[a photo of a hippo]   |
|hen.JPEG         |[a photo of a hen]     |
|ostrich.JPEG     |[a photo of an ostrich]|
|junco.JPEG       |[a photo of a bird]    |
|bluetick.jpg     |[a photo of a dog]     |
|chihuahua.jpg    |[a photo of a dog]     |
|tractor.JPEG     |[a photo of a tractor] |
|ox.JPEG          |[a photo of an ox]     |
+-----------------+-----------------------+

{%- endcapture -%}

{%- capture api_link -%}
[CLIPForZeroShotClassification](/api/com/johnsnowlabs/nlp/annotators/cv/CLIPForZeroShotClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[CLIPForZeroShotClassification](TODO: implement new for new scheme)
[CLIPForZeroShotClassification](/api/python/reference/autosummary/sparknlp/annotator/cv/clip_for_zero_shot_classification/index.html#sparknlp.annotator.cv.clip_for_zero_shot_classification.CLIPForZeroShotClassification)
{%- endcapture -%}

{%- capture source_link -%}
[CLIPForZeroShotClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/CLIPForZeroShotClassification.scala)
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