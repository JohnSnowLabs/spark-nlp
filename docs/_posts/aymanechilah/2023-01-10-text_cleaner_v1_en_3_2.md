---
layout: model
title: Text cleaner v1
author: John Snow Labs
name: text_cleaner_v1
date: 2023-01-10
tags: [en, licensed]
task: OCR Text Cleaner
language: en
nav_key: models
edition: Visual NLP 4.1.0
spark_version: 3.2.1
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Model for cleaning image with text. It is based on text detection model with extra post-processing.

## Predicted Entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/Cards/SparkOcrImageCleaner.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/text_cleaner_v1_en_3.0.0_2.4_1640088709401.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use
<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
pdf_to_image = PdfToImage() \
    .setInputCol("content") \
    .setOutputCol("image") \
    .setResolution(300)

ocr = ImageToText() \
    .setInputCol("image") \
    .setOutputCol("text") \
    .setConfidenceThreshold(70) \
    .setIgnoreResolution(False)

cleaner = ImageTextCleaner \
    .pretrained("text_cleaner_v1", "en", "clinical/ocr") \
    .setInputCol("image") \
    .setOutputCol("corrected_image") \
    .setMedianBlur(0) \
    .setSizeThreshold(10) \
    .setTextThreshold(0.3) \
    .setLinkThreshold(0.2) \
    .setPadding(5) \
    .setBinarize(False)

ocr_corrected = ImageToText() \
    .setInputCol("corrected_image") \
    .setOutputCol("corrected_text") \
    .setConfidenceThreshold(70) \
    .setIgnoreResolution(False)

pipeline = PipelineModel(stages=[
    pdf_to_image,
    ocr,
    cleaner,
    ocr_corrected
])

pdf_example = 'data/pdfs/noised.pdf'
pdf_example_df = spark.read.format("binaryFile").load(pdf_example).cache()

results = pipeline.transform(pdf_example_df).cache()
```
```scala
val pdf_to_image = new PdfToImage() 
    .setInputCol("content") 
    .setOutputCol("image") 
    .setResolution(300)

val ocr = new ImageToText() 
    .setInputCol("image") 
    .setOutputCol("text") 
    .setConfidenceThreshold(70) 
    .setIgnoreResolution(False)

val cleaner = ImageTextCleaner 
    .pretrained("text_cleaner_v1", "en", "clinical/ocr") 
    .setInputCol("image") 
    .setOutputCol("corrected_image") 
    .setMedianBlur(0) 
    .setSizeThreshold(10) 
    .setTextThreshold(0.3) 
    .setLinkThreshold(0.2) 
    .setPadding(5) 
    .setBinarize(False)

val ocr_corrected = new ImageToText() 
    .setInputCol("corrected_image") 
    .setOutputCol("corrected_text") 
    .setConfidenceThreshold(70) 
    .setIgnoreResolution(False)

val pipeline = new PipelineModel().setStages(Array(
    pdf_to_image, 
    ocr, 
    cleaner, 
    ocr_corrected))

val pdf_example = "data/pdfs/noised.pdf"
val pdf_example_df = spark.read.format("binaryFile").load(pdf_example).cache()

val results = pipeline.transform(pdf_example_df).cache()
```
</div>


## Example

{%- capture input_image -%}
![Screenshot](/assets/images/examples_ocr/image4.png)
{%- endcapture -%}

{%- capture output_image -%}
![Screenshot](/assets/images/examples_ocr/image4_out.png)
{%- endcapture -%}


{% include templates/input_output_image.md
input_image=input_image
output_image=output_image
%}


## Output text

```bash
Detected text:
 

 

Sample specifications written by
 , BLEND CASING RECASING

- OLD GOLD STRAIGHT Tobacco Blend

Control for Sample No. 5030

Cigarettes:

OLD GOLD STRAIGHT

 

John H. M. Bohlken

FINAL FLAVOR MENTHOL FLAVOR

Tars and Nicotine, Taste Panel, Burning Time, Gas Phase Analysis,
Benzo (A) Pyrene Analyses — T/C -CF~ O.C S51: Fee -

Written by -- John H. M. Bohlken
Original to -Mr. C. L. Tucker, dr.
Copies to ---Dr. A. W. Spears

C

~
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|text_cleaner_v1|
|Type:|ocr|
|Compatibility:|Visual NLP 4.1.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|77.1 MB|
