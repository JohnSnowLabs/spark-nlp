---
layout: model
title: General model for table detection
author: John Snow Labs
name: general_model_table_detection_v2
date: 2023-01-10
tags: [en, licensed]
task: OCR Table Detection
language: en
edition: Visual NLP 4.1.0
spark_version: 3.2.1
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

For table detection it is proposed the use of CascadeTabNet. It is a Cascade mask Region-based CNN High-Resolution Network (Cascade mask R-CNN HRNet) based model that detects tables on input images. The model is evaluated on ICDAR 2013, ICDAR 2019 and TableBank public datasets. It achieved 3rd rank in ICDAR 2019 post-competition results for table detection while attaining the best accuracy results for the ICDAR 2013 and TableBank dataset.

Here it is used the CascadeTabNet general model for table detection inspired by https://arxiv.org/abs/2004.12629

## Predicted Entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/Cards/SparkOcrImageTableDetection.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/general_model_table_detection_v2_en_3.3.0_3.0_1623301511401.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
binary_to_image = BinaryToImage() \
    .setInputCol("content") \
    .setOutputCol("image") \
    .setImageType(ImageType.TYPE_3BYTE_BGR)

table_detector = ImageTableDetector.pretrained("general_model_table_detection_v2", "en", "clinical/ocr") \
    .setInputCol("image") \
    .setOutputCol("table_regions")

draw_regions = ImageDrawRegions() \
    .setInputCol("image") \
    .setInputRegionsCol("table_regions") \
    .setOutputCol("image_with_regions") \
    .setRectColor(Color.red)

pipeline = PipelineModel(stages=[
    binary_to_image,
    table_detector,
    draw_regions
])

# Download image:
# !wget -q https://github.com/JohnSnowLabs/spark-ocr-workshop/raw/4.0.0-release-candidate/jupyter/data/tab_images/cTDaR_t10168.jpg
imagePath = "cTDaR_t10168.jpg"
image_df = spark.read.format("binaryFile").load(imagePath)

result = pipeline.transform(image_df)
```

```scala
val binary_to_image = new BinaryToImage() 
    .setInputCol("content") 
    .setOutputCol("image") 
    .setImageType(ImageType.TYPE_3BYTE_BGR)

val table_detector = ImageTableDetector.pretrained("general_model_table_detection_v2", "en", "clinical/ocr") \
    .setInputCol("image") 
    .setOutputCol("table_regions")

val draw_regions = new ImageDrawRegions() 
    .setInputCol("image") 
    .setInputRegionsCol("table_regions") 
    .setOutputCol("image_with_regions") 
    .setRectColor(Color.red)

val pipeline = new PipelineModel().setStages(Array(
    binary_to_image, 
    table_detector, 
    draw_regions))

# Download image:
# !wget -q https://github.com/JohnSnowLabs/spark-ocr-workshop/raw/4.0.0-release-candidate/jupyter/data/tab_images/cTDaR_t10168.jpg
val imagePath = "cTDaR_t10168.jpg"
val image_df = spark.read.format("binaryFile").load(imagePath)

val result = pipeline.transform(image_df)
```
</div>



## Example

{%- capture input_image -%}
![Screenshot](/assets/images/examples_ocr/image5.png)
{%- endcapture -%}

{%- capture output_image -%}
![Screenshot](/assets/images/examples_ocr/image5_out.png)
{%- endcapture -%}


{% include templates/input_output_image.md
input_image=input_image
output_image=output_image
%}


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ocr_table_detection_general_model|
|Type:|ocr|
|Compatibility:|Visual NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Output Labels:|[table regions]|
|Language:|en|

## Benchmarking

```bash
3rd rank in ICDAR 2019 post-competition
1st rank in ICDAR 2013 and TableBank dataset
```