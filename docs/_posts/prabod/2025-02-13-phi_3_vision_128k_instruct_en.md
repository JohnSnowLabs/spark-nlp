---
layout: model
title: Phi-3-vision-128k-instruct
author: John Snow Labs
name: phi_3_vision_128k_instruct
date: 2025-02-13
tags: [en, open_source, openvino, phi3v]
task: Image Captioning
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: Phi3Vision
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The Phi-3-Vision-128K-Instruct is a lightweight, state-of-the-art open multimodal model built upon datasets which include - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data both on text and vision. The model belongs to the Phi-3 model family, and the multimodal version comes with 128K context length (in tokens) it can support. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/phi_3_vision_128k_instruct_en_5.5.1_3.0_1739416284436.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/phi_3_vision_128k_instruct_en_5.5.1_3.0_1739416284436.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
url1 = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
url2 = "http://images.cocodataset.org/val2017/000000039769.jpg"

Path("images").mkdir(exist_ok=True)

!wget -q -O images/image1.jpg {url1}
!wget -q -O images/image2.jpg {url2}



images_path = "file://" + os.getcwd() + "/images/"
image_df = spark.read.format("image").load(
    path=images_path
)

test_df = image_df.withColumn("text", lit("<|user|> \n <|image_1|> \n What's this picture about? <|end|>\n <|assistant|>\n"))

image_assembler = ImageAssembler().setInputCol("image").setOutputCol("image_assembler")

imageClassifier = Phi3Vision.pretrained("phi_3_vision_128k_instruct","en")\
            .setMaxOutputLength(50) \
            .setInputCols("image_assembler") \
            .setOutputCol("answer")

pipeline = Pipeline(
            stages=[
                image_assembler,
                imageClassifier,
            ]
        )

results = pipeline.fit(test_df).transform(test_df)
```
```scala
val imageFolder = "src/test/resources/image/"
    val imageDF: DataFrame = ResourceHelper.spark.read
      .format("image")
      .option("dropInvalid", value = true)
      .load(imageFolder)

    val testDF: DataFrame = imageDF.withColumn(
      "text",
      lit("<|user|> \n <|image_1|> \n What's this picture about? <|end|>\n <|assistant|>\n"))

val imageAssembler: ImageAssembler = new ImageAssembler()
      .setInputCol("image")
      .setOutputCol("image_assembler")

val loadModel = Phi3Vision
      .pretrained("phi_3_vision_128k_instruct","en")
      .setInputCols("image_assembler")
      .setOutputCol("answer")
      .setMaxOutputLength(50)

    val newPipeline: Pipeline =
      new Pipeline().setStages(Array(imageAssembler, loadModel))

    newPipeline.fit(testDF).transform(testDF).show()
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|phi_3_vision_128k_instruct|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|3.3 GB|

## References

https://huggingface.co/microsoft/Phi-3-vision-128k-instruct