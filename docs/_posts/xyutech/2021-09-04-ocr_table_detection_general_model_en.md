---
layout: model
title: General model for table detection
author: John Snow Labs
name: ocr_table_detection_general_model
date: 2021-09-04
tags: [en, licensed]
task: OCR Table Detection & Recognition
language: en
edition: Visual NLP 3.0.0
spark_version: 3.0
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
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/tutorials/Certification_Trainings/others/SparkOcrImageTableDetection.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/ocr_table_detection_general_model_en_3.0.0_3.0_1630757579641.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    from pyspark.ml import PipelineModel
    from sparkocr.transformers import *
    
    imagePath = "path to image"
    image_df = spark.read.format("binaryFile").load(imagePath)

    binary_to_image = BinaryToImage() 
    binary_to_image.setImageType(ImageType.TYPE_3BYTE_BGR)
    
    table_detector = ImageTableDetector.pretrained("general_model_table_detection_v2", "en", "clinical/ocr")
    table_detector.setInputCol("image")
    table_detector.setOutputCol("table_regions")
    
    draw_regions = ImageDrawRegions()
    draw_regions.setInputCol("image")
    draw_regions.setInputRegionsCol("table_regions")
    draw_regions.setOutputCol("image_with_regions")
    draw_regions.setRectColor(Color.red)
    
    pipeline = PipelineModel(stages=[
        binary_to_image,
        table_detector,
        draw_regions
    ])

    result = pipeline.transform(image_df)
```
```scala
import com.johnsnowlabs.ocr.transformers.*
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"
var imgDf = spark.read.format("binaryFile").load(imagePath)

var bin2imTransformer = new BinaryToImage()
bin2imTransformer.setImageType(ImageType.TYPE_3BYTE_BGR)

val dataFrame = bin2imTransformer.transform(imgDf)
val tableDetector = ImageTableDetector
.pretrained("general_model_table_detection_v2", "en", "clinical/ocr")
.setInputCol("image")
.setOutputCol("table regions")

val results = pipeline
  .fit(imgDf)
  .transform(imgDf)
  .select("label", "exception")
  .cache()
```
</div>



## Example

### Input:
![Screenshot](docs/_examples_ocr/image5.png)

### Output:
![Screenshot](docs/_examples_ocr/image5_out.png)


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
1st rank in  ICDAR 2013 and TableBank dataset
```