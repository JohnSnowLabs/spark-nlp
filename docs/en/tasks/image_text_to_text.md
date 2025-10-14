---
layout: docs
header: true
seotitle:
title: Image Text to Text
permalink: docs/en/tasks/image_text_to_text
key: docs-tasks-image-text-to-text
modify_date: "2025-08-23"
show_nav: true
sidebar:
  nav: sparknlp
---

**Image-text-to-text** models take in an image and text prompt and output text. These models are also called **vision-language models (VLMs)**. The difference from image-to-text models is that these models take an additional text input, not restricting the model to certain use cases like image captioning, and may also be trained to accept a conversation as input.

### Types of Vision Language Models  

- **Base Models**  
  Pre-trained on large datasets and designed to be fine-tuned for specific tasks, such as Google’s [PaliGemma](https://sparknlp.org/models?q=PaliGemma&sort=downloads&annotator=PaliGemmaForMultiModal&type=model) family.  

- **Instruction-tuned Models**  
  Base models fine-tuned to better follow written instructions, like [Qwen2.5-7B-Instruct](https://sparknlp.org/models?q=Qwen2.5-7B-Instruct&sort=downloads).  

- **Conversational Models**  
  Base models fine-tuned on dialogue data to handle multi-turn conversations, such as [DeepSeek-VL-7B-Chat](https://sparknlp.org/models?q=deepseek-vl-7b-chat&type=model&sort=downloads).  

### Common Use Cases  

- **Multimodal Dialogue**
These models can act as assistants that handle both text and images in a conversation. They remember the context and can respond across multiple turns while referring back to the same image.

- **Object Detection and Image Segmentation**
Some models can find, outline, or locate objects in an image. For example, you could ask if one object is behind another. They can even provide bounding boxes or segmentation masks directly, unlike older models that were trained only for detection or segmentation.

- **Visual Question Answering**
By learning from image–text pairs, these models can answer questions about an image or create captions that describe it.

- **Document Question Answering and Retrieval**
Documents often include tables, charts, and images. Instead of using OCR, these models can read documents directly and pull out the needed information.

- **Image Recognition with Instructions**
If you give a model detailed descriptions, it can identify or classify specific things in an image, rather than being limited to fixed label sets. For example, instead of just labeling “dog” or “cat,” you could ask it to find “a small brown dog wearing a red collar,” and it would pick out exactly that.

## How to use

![Cat In A Box](https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp/master/src/test/resources/images/image1.jpg)

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
from sparknlp.base import DocumentAssembler, ImageAssembler
from sparknlp.annotator import AutoGGUFVisionModel
from pyspark.sql.functions import lit
from pyspark.ml import Pipeline

images_path = "path/to/images/folder"
prompt = "Describe this image."

data = ImageAssembler.loadImagesAsBytes(spark, images_path)
data = data.withColumn("prompt", lit(prompt))

document_assembler = (
    DocumentAssembler()
    .setInputCol("prompt")
    .setOutputCol("document")
)

image_assembler = (
    ImageAssembler()
    .setInputCol("image")
    .setOutputCol("image_assembler")
)

auto_gguf_vision_model = (
    AutoGGUFVisionModel.pretrained("qwen2_vl_2b_instruct_q4_gguf")
    .setInputCols(["document", "image_assembler"])
    .setOutputCol("completions")
)

pipeline = Pipeline(stages=[
    document_assembler,
    image_assembler,
    auto_gguf_vision_model
])

model = pipeline.fit(data)
result = model.transform(data)

result.selectExpr(
    "reverse(split(image.origin, '/'))[0] as image_name",
    "completions.result"
).show(truncate=False)

```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.sql.functions.lit
import org.apache.spark.ml.Pipeline

val imagesPath = "path/to/images/folder"
val prompt = "Describe this image."

var data = ImageAssembler.loadImagesAsBytes(spark, imagesPath).withColumn("prompt", lit(prompt))

val documentAssembler = new DocumentAssembler()
  .setInputCol("prompt")
  .setOutputCol("document")

val imageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val autoGgufVisionModel = AutoGGUFVisionModel.pretrained("qwen2_vl_2b_instruct_q4_gguf")
  .setInputCols(Array("document", "image_assembler"))
  .setOutputCol("completions")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  imageAssembler,
  autoGgufVisionModel
))

val model = pipeline.fit(data)
val result = model.transform(data)

result.selectExpr(
  "reverse(split(image.origin, '/'))[0] as image_name",
  "completions.result"
).show(truncate = false)

```
</div>

<div class="tabs-box" markdown="1">
```
+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|image_name      |result                                                                                                                                                                                                                                                                    |
+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|cat_in_a_box.jpg|[The image shows a fluffy gray cat lying inside an open cardboard box on a carpeted floor. The cat appears to be relaxed and is stretched out in a comfortable position, with its paws sticking out of the box. In the background, there is a white couch against a wall.]|
+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```
</div>

## Useful Resources

- [Vision Language Models (Better, Faster, Stronger)](https://huggingface.co/blog/vlms-2025)
- [Vision Language Models Explained](https://huggingface.co/blog/vlms)
- [Welcome PaliGemma 2 – New vision language models by Google](https://huggingface.co/blog/paligemma2)
- [Multimodal RAG using ColPali and Qwen2-VL](https://github.com/merveenoyan/smol-vision/blob/main/ColPali_%2B_Qwen2_VL.ipynb)
- [Preference Optimization for Vision Language Models with TRL](https://huggingface.co/blog/dpo_vlm)