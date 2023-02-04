---
layout: model
title: Ocr pipeline with Rest-Api
author: John Snow Labs
name: ocr_restapi
date: 2023-01-03
tags: [en, licensed, ocr, RestApi]
task: Ocr RestApi
language: en
edition: Visual NLP 4.0.0
spark_version: 3.2.1
supported: true
annotator: OcrRestApi
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

RestAPI pipeline implementation for the OCR task, using tesseract models. Tesseract is an Optical Character Recognition (OCR) engine developed by Google. It is an open-source tool that can be used to recognize text in images and convert it into machine-readable text. The engine is based on a neural network architecture and uses machine learning algorithms to improve its accuracy over time.

Tesseract has been trained on a variety of datasets to improve its recognition capabilities. These datasets include images of text in various languages and scripts, as well as images with different font styles, sizes, and orientations. The training process involves feeding the engine with a large number of images and their corresponding text, allowing the engine to learn the patterns and characteristics of different text styles. One of the most important datasets used in training Tesseract is the UNLV dataset, which contains over 400,000 images of text in different languages, scripts, and font styles. This dataset is widely used in the OCR community and has been instrumental in improving the accuracy of Tesseract. Other datasets that have been used in training Tesseract include the ICDAR dataset, the IIIT-HWS dataset, and the RRC-GV-WS dataset.

In addition to these datasets, Tesseract also uses a technique called adaptive training, where the engine can continuously improve its recognition capabilities by learning from new images and text. This allows Tesseract to adapt to new text styles and languages, and improve its overall accuracy.

## Predicted Entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/tutorials/Certification_Trainings/6.2.SparkOcrRestApi.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
<button class="button button-orange" disabled>Download</button>

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
binary_to_image = BinaryToImage() \
    .setInputCol("content") \
    .setOutputCol("image") \
    .setImageType(ImageType.TYPE_3BYTE_BGR)

ocr = ImageToText() \
    .setInputCol("image") \
    .setOutputCol("text")

pipeline = PipelineModel(stages=[
    binary_to_image,
    ocr
])

## Start server
SERVER_HOST = "localhost"
SERVER_PORT = 8889
SERVER_API_NAME = "spark_ocr_api"

checkpoint_dir = tempfile.TemporaryDirectory("_spark_ocr_server_checkpoint")
df = spark.readStream.server() \
    .address(SERVER_HOST, SERVER_PORT, SERVER_API_NAME) \
    .load() \
    .parseRequest(SERVER_API_NAME, schema=StructType().add("image", BinaryType())) \
    .withColumn("path", f.lit("")) \
    .withColumnRenamed("image", "content")

replies = pipeline.transform(df)\
    .makeReply("text") 

server = replies\
    .writeStream \
    .server() \
    .replyTo(SERVER_API_NAME) \
    .queryName("spark_ocr") \
    .option("checkpointLocation", checkpoint_dir) \
    .start()

## Call API
imagePath = pkg_resources.resource_filename('sparkocr', '/resources/ocr/images/check.jpg')
with open(imagePath, "rb") as image_file:
    im_bytes = image_file.read()

im_b64 = base64.b64encode(im_bytes).decode("utf8")
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
payload = json.dumps({"image": im_b64})

r = requests.post(data=payload, headers=headers, url=f"http://{SERVER_HOST}:{SERVER_PORT}/{SERVER_API_NAME}")
```

</div>

## Example

### Input:
![Screenshot](/assets/images/examples_ocr/image2.png)

## Output text

```bash
Response:

STARBUCKS Store #19208
11902 Euclid Avenue
Cleveland, OH (216) 229-U749

CHK 664250
12/07/2014 06:43 PM
112003. Drawers 2. Reg: 2

¥t Pep Mocha 4.5
Sbux Card 495
AMXARKERARANG 228
Subtotal $4.95
Total $4.95
Change Cue BO LOO
- Check Closed ~

"49/07/2014 06:43 py

oBUX Card «3228 New Balance: 37.45
Card is registertd
```
