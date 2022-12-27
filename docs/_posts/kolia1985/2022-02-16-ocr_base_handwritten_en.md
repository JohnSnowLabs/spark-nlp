---
layout: model
title: Оcr base for handwritten text
author: John Snow Labs
name: ocr_base_handwritten
date: 2022-02-16
tags: [en, licensed]
task: OCR Text Detection & Recognition
language: en
edition: Visual NLP 3.3.3
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Ocr base model for recognise handwritten text based on TrOcr architecture.  The TrOCR model was proposed in TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei. TrOCR consists of an image Transformer encoder and an autoregressive text Transformer decoder to perform optical character recognition (OCR).  The abstract from the paper is the following:  Text recognition is a long-standing research problem for document digitalization. Existing approaches for text recognition are usually built based on CNN for image understanding and RNN for char-level text generation. In addition, another language model is usually needed to improve the overall accuracy as a post-processing step. In this paper, we propose an end-to-end text recognition approach with pre-trained image Transformer and text Transformer models, namely TrOCR, which leverages the Transformer architecture for both image understanding and wordpiece-level text generation. The TrOCR model is simple but effective, and can be pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. Experiments show that the TrOCR model outperforms the current state-of-the-art models on both printed and handwritten text recognition tasks.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/TrainingNotebooks/tutorials/Certification_Trainings/1.4.Handwritten_Text_Recognition.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/ocr_base_handwritten_en_3.3.3_2.4_1645034046021.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use
<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    from pyspark.ml import PipelineModel
    from sparkocr.transformers import *
    
    imagePath = "path to image"
    image_df = spark.read.format("binaryFile").load(imagePath)

    binary_to_image = BinaryToImage() 
    
    text_detector = ImageTextDetectorV2 \
        .pretrained("image_text_detector_v2", "en", "clinical/ocr") \
        .setInputCol("image") \
        .setOutputCol("text_regions") \
        .setWithRefiner(True) \
        .setSizeThreshold(10) \
        .setScoreThreshold(0.2) \
        .setTextThreshold(0.2) \
        .setLinkThreshold(0.3) \
        .setWidth(500)
    
    ocr = ImageToTextV2.pretrained("ocr_base_handwritten", "en", "clinical/ocr") \
        .setInputCols(["image", "text_regions"]) \
        .setGroupImages(True) \
        .setOutputCol("text")
    
    draw_regions = ImageDrawRegions() \
        .setInputCol("image") \
        .setInputRegionsCol("text_regions") \
        .setOutputCol("image_with_regions") \
        .setRectColor(Color.green) \
        .setRotated(True)
    
    pipeline = PipelineModel(stages=[
        binary_to_image,
        text_detector,
        ocr,
        draw_regions
    ])

    result = pipeline.transform(image_df)
    print(("").join([x.text for x in result.select("text").collect()]))
```
```scala

```
</div>


## Result

```bash
This is an example of handwritten
bext
Let's # check the performance!
" hope it will be awesome
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ocr_base_handwritten|
|Type:|ocr|
|Compatibility:|Visual NLP 3.3.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|781.9 MB|
