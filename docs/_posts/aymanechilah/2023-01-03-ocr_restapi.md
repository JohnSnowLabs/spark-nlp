---
layout: model
title: ocr_restapi
author: John Snow Labs
name: ocr_restapi
date: 2023-01-03
tags: [en, licensed]
task: Ocr RestApi
language: en
edition: Visual NLP 3.14.0
spark_version: 3.0
supported: true
annotator: OcrRestApi
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

RestAPI pipeline implementation for the OCR task, using tesseract models. Tesseract is an open source text recognition (OCR) Engine, available under the Apache 2.0 license. Library pros are trainedlanguage models (>192), different kinds of recognition (image as word, text block, vertical text)


## Predicted Entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/tutorials/Certification_Trainings/6.2.SparkOcrRestApi.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
<button class="button button-orange" disabled>Download</button>

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
    
    from pyspark.ml import PipelineModel
    from sparkocr.transformers import *
    
    imagePath = "path to image"
    bin_df = spark.read.format("binaryFile").load(imagePath)
    
    binary_to_image = BinaryToImage() 
    
    ocr = ImageToText() \
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
    with open(imagePath, "rb") as image_file:
        im_bytes = image_file.read()

    im_b64 = base64.b64encode(im_bytes).decode("utf8")
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    payload = json.dumps({"image": im_b64})

    r = requests.post(data=payload, headers=headers, url=f"http://{SERVER_HOST}:{SERVER_PORT}/{SERVER_API_NAME}")
```
```scala
```
</div>

## Example

### Input:
![Screenshot](docs/_examples_ocr/image2.png)

### Output:

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