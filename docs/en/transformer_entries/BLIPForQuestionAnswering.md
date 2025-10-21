{%- capture title -%}
BLIPForQuestionAnswering
{%- endcapture -%}

{%- capture description -%}
BLIP (Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation) Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text decoder. The vision encoder will encode the input image, the text encoder will encode the input question together with the encoding of the image, and the text decoder will output the answer to the question.

Pretrained models can be loaded with `pretrained` of the companion object:
```scala
val loadModel = BLIPForQuestionAnswering.pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("answer")
```
The default model is `"blip_vqa_base"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?task=Question+Answering&annotator=BLIPForQuestionAnswering).

Spark NLP also supports Hugging Face transformer-based code generation models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Sources** :

- [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)
- [BLIP on GitHub](https://github.com/salesforce/BLIP)

**Paper abstract**

*Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to video-language tasks in a zero-shot manner. Code, models, and datasets are released at this https URL.*
{%- endcapture -%}

{%- capture input_anno -%}
IMAGE_ASSEMBLER
{%- endcapture -%}

{%- capture output_anno -%}
ANSWER
{%- endcapture -%}

{%- capture api_link -%}
[BLIPForQuestionAnswering](/api/com/johnsnowlabs/nlp/annotators/cv/BLIPForQuestionAnswering.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[BLIPForQuestionAnswering](/api/python/reference/autosummary/sparknlp/annotator/cv/blip_for_question_answering/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[BLIPForQuestionAnswering](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/BLIPForQuestionAnswering.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import ImageAssembler
from sparknlp.annotator import BLIPForQuestionAnswering
from pyspark.sql.functions import lit
from pyspark.ml import Pipeline

# To proceed please create a DataFrame with two required columns:
# - 'image': contains file paths of input images
# - 'text' : contains the visual question to be asked about each image

images_path = "./images/"
image_df = spark.read.format("image").load(images_path)

question_prompt = "What's this picture about?"
input_df = image_df.withColumn("text", lit(question_prompt))

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

vqa_model = BLIPForQuestionAnswering.pretrained() \
    .setInputCols(["image_assembler", "text"]) \
    .setOutputCol("answer") \
    .setSize(384)

pipeline = Pipeline(stages=[
    image_assembler,
    vqa_model
])

fitted_model = pipeline.fit(input_df)
result_df = fitted_model.transform(input_df)

result_df.select("image_assembler.origin", "answer.result").show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.ImageAssembler
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.lit

// To proceed please create a DataFrame with two required columns:
// - 'image': contains file paths of input images
// - 'text' : contains the visual question to be asked about each image

val imagesPath = "./images/"
val imageDF = spark.read.format("image").load(imagesPath)

val questionPrompt = "What's this picture about?"
val inputDF = imageDF.withColumn("text", lit(questionPrompt))

val imageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val vqaModel = BLIPForQuestionAnswering.pretrained()
  .setInputCols("image_assembler", "text")
  .setOutputCol("answer")
  .setSize(384)

val pipeline = new Pipeline().setStages(Array(
  imageAssembler,
  vqaModel
))

val fittedModel = pipeline.fit(inputDF)
val resultDF = fittedModel.transform(inputDF)

resultDF.select("image_assembler.origin", "answer.result").show(false)
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