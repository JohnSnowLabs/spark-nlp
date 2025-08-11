---
layout: model
title: Qwen2-VL-2B-Instruct (Q4 GGUF Quantized)
author: John Snow Labs
name: qwen2_vl_2b_instruct_q4_gguf
date: 2025-08-11
tags: [qwen2_vl, image_to_text, multimodal, conversational, instruct, q4, 2b, en, open_source, llamacpp]
task: Image Captioning
language: en
edition: Spark NLP 6.1.1
spark_version: 3.0
supported: true
engine: llamacpp
annotator: AutoGGUFVisionModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Qwen2-VL-2B-Instruct is a 2-billion-parameter vision-language model fine-tuned for following instructions across text, image, and video inputs, enabling tasks like captioning, visual question answering, and multimodal dialogue.

Originally from [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/qwen2_vl_2b_instruct_q4_gguf_en_6.1.1_3.0_1754924858631.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/qwen2_vl_2b_instruct_q4_gguf_en_6.1.1_3.0_1754924858631.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import DocumentAssembler, ImageAssembler
from sparknlp.annotator import AutoGGUFVisionModel
from pyspark.sql.functions import lit
from pyspark.ml import Pipeline

images_path = "path/to/images/folder"
prompt = "Caption this image."

data = ImageAssembler.loadImagesAsBytes(spark, images_path)
data = data.withColumn("caption", lit(prompt))

document_assembler = (
    DocumentAssembler()
    .setInputCol("caption")
    .setOutputCol("caption_document")
)

image_assembler = (
    ImageAssembler()
    .setInputCol("image")
    .setOutputCol("image_assembler")
)

qwen_chat_template = """<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

autoGGUFVisionModel = (
    AutoGGUFVisionModel.pretrained("qwen2_vl_2b_instruct_q4_gguf")
    .setInputCols(["caption_document", "image_assembler"])
    .setOutputCol("completions")
    .setChatTemplate(qwen_chat_template)
    .setBatchSize(4)
    .setNGpuLayers(32)
    .setNCtx(4096)
    .setMinKeep(0)
    .setMinP(0.05)
    .setNPredict(64)
    .setNProbs(0)
    .setPenalizeNl(False)
    .setRepeatLastN(256)
    .setRepeatPenalty(1.1)
    .setStopStrings(["</s>", "<|im_end|>", "User:"])
    .setTemperature(0.2)
    .setTfsZ(1)
    .setTypicalP(1)
    .setTopK(40)
    .setTopP(0.95)
)

pipeline = Pipeline().setStages([
    document_assembler,
    image_assembler,
    autoGGUFVisionModel
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

val images_path = "path/to/images/folder"
val prompt = "Caption this image."

var data = ImageAssembler.loadImagesAsBytes(spark, images_path)
data = data.withColumn("caption", lit(prompt))

val document_assembler = new DocumentAssembler()
  .setInputCol("caption")
  .setOutputCol("caption_document")

val image_assembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val qwen_chat_template = """<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

val autoGGUFVisionModel = AutoGGUFVisionModel.pretrained("qwen2_vl_2b_instruct_q4_gguf")
  .setInputCols(Array("caption_document", "image_assembler"))
  .setOutputCol("completions")
  .setChatTemplate(qwen_chat_template)
  .setBatchSize(4)
  .setNGpuLayers(32)
  .setNCtx(4096)
  .setMinKeep(0)
  .setMinP(0.05)
  .setNPredict(64)
  .setNProbs(0)
  .setPenalizeNl(false)
  .setRepeatLastN(256)
  .setRepeatPenalty(1.1)
  .setStopStrings(Array("</s>", "<|im_end|>", "User:"))
  .setTemperature(0.2)
  .setTfsZ(1)
  .setTypicalP(1)
  .setTopK(40)
  .setTopP(0.95)

val pipeline = new Pipeline().setStages(Array(
  document_assembler,
  image_assembler,
  autoGGUFVisionModel
))

val model = pipeline.fit(data)
val result = model.transform(data)

result.selectExpr(
  "reverse(split(image.origin, '/'))[0] as image_name",
  "completions.result"
).show(false)

```
</div>

## Results

```bash

+---------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
|image_name                                   |result                                                                                                                                   |
+---------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
|[prescription_02.png, images, content, file:]|["Outpatient Summary: Rheumatology Consultation for Systemic Lupus Erythematosus and Scleroderma Overlap with Interstitial Lung Disease"]|
|[prescription_01.png, images, content, file:]|["Medical prescription for treatment of fever and headache with medication details."]                                                    |
+---------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|qwen2_vl_2b_instruct_q4_gguf|
|Compatibility:|Spark NLP 6.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[caption_document, image_assembler]|
|Output Labels:|[completions]|
|Language:|en|
|Size:|1.6 GB|