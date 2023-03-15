---
layout: model
title: Оcr small for printed text
author: John Snow Labs
name: ocr_small_printed
date: 2023-01-10
tags: [en, licensed]
task: OCR Text Detection & Recognition
language: en
nav_key: models
edition: Visual NLP 3.3.3
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Ocr small model for recognise printed text based on TrOcr architecture. The TrOCR model was proposed in TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei. TrOCR consists of an image Transformer encoder and an autoregressive text Transformer decoder to perform optical character recognition (OCR).  The abstract from the paper is the following:  Text recognition is a long-standing research problem for document digitalization. Existing approaches for text recognition are usually built based on CNN for image understanding and RNN for char-level text generation. In addition, another language model is usually needed to improve the overall accuracy as a post-processing step. In this paper, we propose an end-to-end text recognition approach with pre-trained image Transformer and text Transformer models, namely TrOCR, which leverages the Transformer architecture for both image understanding and wordpiece-level text generation. The TrOCR model is simple but effective, and can be pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. Experiments show that the TrOCR model outperforms the current state-of-the-art models on both printed and handwritten text recognition tasks.

## Predicted Entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/tutorials/Certification_Trainings/1.3.Trasformer_based_Text_Recognition.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/ocr_small_printed_en_3.3.3_2.4_1645007455031.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
binary_to_image = BinaryToImage() \
    .setInputCol("content") \
    .setOutputCol("image") \
    .setImageType(ImageType.TYPE_3BYTE_BGR)

text_detector = ImageTextDetectorV2 \
    .pretrained("image_text_detector_v2", "en", "clinical/ocr") \
    .setInputCol("image") \
    .setOutputCol("text_regions") \
    .setWithRefiner(False) \
    .setSizeThreshold(15) \
    .setLinkThreshold(0.3)

ocr = ImageToTextV2.pretrained("ocr_small_printed", "en", "clinical/ocr") \
    .setInputCols(["image", "text_regions"]) \
    .setOutputCol("hocr") \
    .setOutputFormat(OcrOutputFormat.HOCR) \
    .setGroupImages(False) 

tokenizer = HocrTokenizer() \
    .setInputCol("hocr") \
    .setOutputCol("token") \

draw_annotations = ImageDrawAnnotations() \
    .setInputCol("image") \
    .setInputChunksCol("token") \
    .setOutputCol("image_with_annotations") \
    .setFilledRect(False) \
    .setFontSize(14) \
    .setRectColor(Color.red)

pipeline = PipelineModel(stages=[
    binary_to_image,
    text_detector,
    ocr,
    tokenizer,
    draw_annotations
])

image_path = pkg_resources.resource_filename('sparkocr', 'resources/ocr/images/check.jpg')
image_example_df = spark.read.format("binaryFile").load(image_path)

result = pipeline.transform(image_example_df).cache()
```
```scala
val binary_to_image = new BinaryToImage() 
    .setInputCol("content") 
    .setOutputCol("image") 
    .setImageType(ImageType.TYPE_3BYTE_BGR)

val text_detector = ImageTextDetectorV2.pretrained("image_text_detector_v2", "en", "clinical/ocr") 
    .setInputCol("image") 
    .setOutputCol("text_regions") 
    .setWithRefiner(False) 
    .setSizeThreshold(15) 
    .setLinkThreshold(0.3)

val ocr = ImageToTextV2.pretrained("ocr_small_printed", "en", "clinical/ocr") 
    .setInputCols("image", "text_regions") 
    .setOutputCol("hocr") 
    .setOutputFormat(OcrOutputFormat.HOCR) 
    .setGroupImages(False) 

val tokenizer = new HocrTokenizer() 
    .setInputCol("hocr") 
    .setOutputCol("token") 

val draw_annotations = new ImageDrawAnnotations() 
    .setInputCol("image") 
    .setInputChunksCol("token") 
    .setOutputCol("image_with_annotations") 
    .setFilledRect(False) 
    .setFontSize(14) 
    .setRectColor(Color.red)

val pipeline = new PipelineModel().setStages(Array(
    binary_to_image, 
    text_detector, 
    ocr, 
    tokenizer, 
    draw_annotations))

val image_path = pkg_resources.resource_filename("sparkocr", "resources/ocr/images/check.jpg"")
val image_example_df = spark.read.format("binaryFile").load(image_path)

val result = pipeline.transform(image_example_df).cache()
```
</div>

## Example

{%- capture input_image -%}
![Screenshot](/assets/images/examples_ocr/image2.png)
{%- endcapture -%}

{%- capture output_image -%}
![Screenshot](/assets/images/examples_ocr/image2_out.png)
{%- endcapture -%}


{% include templates/input_output_image.md
input_image=input_image
output_image=output_image
%}

## Output text

```bash
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
  <head>
    <title></title>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
    <meta name='ocr-system' content='Spark OCR' />
    <meta name='ocr-langs' content='en' />
    <meta name='ocr-number-of-pages' content='1' />
    <meta name='ocr-capabilities' content='ocr_page ocr_carea ocr_line ocrx_word ocrp_lang'/>
  </head>
  <body>
    <div class='ocr_page' lang='en' title='bbox 0 0 553 715'>
        <div class='ocr_carea' lang='en' title='bbox 18 29 553 715'>
            <span class='ocr_line' id='line_0' title='bbox 139 29 437 58; baseline 0 -5'>
                <span class='ocrx_word' id='word_0_0' title='bbox 139 29 265 56'>STARBUCKS</span>
                <span class='ocrx_word' id='word_0_1' title='bbox 273 30 344 58'>STORE</span>
                <span class='ocrx_word' id='word_0_2' title='bbox 353 30 437 58'>#10208</span>
            </span>
            <span class='ocr_line' id='line_1' title='bbox 158 62 409 91; baseline 0 -5'>
                <span class='ocrx_word' id='word_1_0' title='bbox 158 62 225 90'>11302</span>
                <span class='ocrx_word' id='word_1_1' title='bbox 234 63 318 90'>EUCLID</span>
                <span class='ocrx_word' id='word_1_2' title='bbox 326 66 409 91'>AVENUE</span>
            </span>
            <span class='ocr_line' id='line_2' title='bbox 99 95 477 125; baseline 0 -5'>
                <span class='ocrx_word' id='word_2_0' title='bbox 99 95 226 123'>CLEVELAND</span>
                <span class='ocrx_word' id='word_2_1' title='bbox 246 97 277 123'>OH</span>
                <span class='ocrx_word' id='word_2_2' title='bbox 288 96 355 125'>(216)</span>
                <span class='ocrx_word' id='word_2_3' title='bbox 367 97 477 124'>229-0749</span>
            </span>
            <span class='ocr_line' id='line_3' title='bbox 219 162 358 189; baseline 0 -5'>
                <span class='ocrx_word' id='word_3_0' title='bbox 219 162 263 188'>CHK</span>
                <span class='ocrx_word' id='word_3_1' title='bbox 272 162 358 189'>664290</span>
            </span>
            <span class='ocr_line' id='line_4' title='bbox 155 194 410 223; baseline 0 -5'>
                <span class='ocrx_word' id='word_4_0' title='bbox 155 194 289 223'>12/07/2014</span>
                <span class='ocrx_word' id='word_4_1' title='bbox 298 196 372 223'>06:43</span>
                <span class='ocrx_word' id='word_4_2' title='bbox 380 196 410 222'>PM</span>
            </span>
            <span class='ocr_line' id='line_5' title='bbox 102 226 462 258; baseline 0 -5'>
                <span class='ocrx_word' id='word_5_0' title='bbox 102 226 198 254'>1912003</span>
                <span class='ocrx_word' id='word_5_1' title='bbox 232 230 329 255'>DRAMER:</span>
                <span class='ocrx_word' id='word_5_2' title='bbox 340 230 356 254'>2</span>
                <span class='ocrx_word' id='word_5_3' title='bbox 378 231 435 258'>REG:</span>
                <span class='ocrx_word' id='word_5_4' title='bbox 447 231 462 254'>2</span>
            </span>
            <span class='ocr_line' id='line_6' title='bbox 60 292 489 321; baseline 0 -5'>
                <span class='ocrx_word' id='word_6_0' title='bbox 60 294 90 319'>VT</span>
                <span class='ocrx_word' id='word_6_1' title='bbox 98 294 144 321'>PEP</span>
                <span class='ocrx_word' id='word_6_2' title='bbox 152 292 223 320'>MOCHA</span>
                <span class='ocrx_word' id='word_6_3' title='bbox 433 294 489 319'>4.95</span>
            </span>
            <span class='ocr_line' id='line_7' title='bbox 59 326 489 353; baseline 0 -5'>
                <span class='ocrx_word' id='word_7_0' title='bbox 59 326 116 351'>SBUX</span>
                <span class='ocrx_word' id='word_7_1' title='bbox 126 327 184 353'>CARD</span>
                <span class='ocrx_word' id='word_7_2' title='bbox 434 328 489 353'>4.95</span>
            </span>
            <span class='ocr_line' id='line_8' title='bbox 59 358 276 385; baseline 0 -5'>
                <span class='ocrx_word' id='word_8_0' title='bbox 59 358 276 385'>XXXXXXXXXXXX3228</span>
            </span>
            <span class='ocr_line' id='line_9' title='bbox 60 424 489 454; baseline 0 -5'>
                <span class='ocrx_word' id='word_9_0' title='bbox 60 424 168 451'>SUBTOTAL</span>
                <span class='ocrx_word' id='word_9_1' title='bbox 419 426 489 454'>$4.95</span>
            </span>
            <span class='ocr_line' id='line_10' title='bbox 60 456 489 487; baseline 0 -5'>
                <span class='ocrx_word' id='word_10_0' title='bbox 60 456 127 484'>TOTAL</span>
                <span class='ocrx_word' id='word_10_1' title='bbox 418 459 489 487'>$4.95</span>
            </span>
            <span class='ocr_line' id='line_11' title='bbox 19 492 553 519; baseline 0 -5'>
                <span class='ocrx_word' id='word_11_0' title='bbox 19 492 180 519'>CHANGE</span>
                <span class='ocrx_word' id='word_11_1' title='bbox 205 492 287 517'>DUE</span>
                <span class='ocrx_word' id='word_11_2' title='bbox 419 492 553 518'>$0.00</span>
            </span>
            <span class='ocr_line' id='line_12' title='bbox 179 556 342 583; baseline 0 -5'>
                <span class='ocrx_word' id='word_12_0' title='bbox 179 556 249 583'>CHECK</span>
                <span class='ocrx_word' id='word_12_1' title='bbox 259 556 342 583'>CLOSED</span>
            </span>
            <span class='ocr_line' id='line_13' title='bbox 154 589 410 618; baseline 0 -5'>
                <span class='ocrx_word' id='word_13_0' title='bbox 154 589 289 618'>12/07/2014</span>
                <span class='ocrx_word' id='word_13_1' title='bbox 297 591 369 617'>06:43</span>
                <span class='ocrx_word' id='word_13_2' title='bbox 378 590 410 617'>PM</span>
            </span>
            <span class='ocr_line' id='line_14' title='bbox 18 654 489 684; baseline 0 -5'>
                <span class='ocrx_word' id='word_14_0' title='bbox 18 654 76 681'>SBUX</span>
                <span class='ocrx_word' id='word_14_1' title='bbox 84 656 142 680'>CARD</span>
                <span class='ocrx_word' id='word_14_2' title='bbox 153 654 223 682'>X3228</span>
                <span class='ocrx_word' id='word_14_3' title='bbox 232 657 275 681'>NEW</span>
                <span class='ocrx_word' id='word_14_4' title='bbox 283 657 393 684'>BALANCE:</span>
                <span class='ocrx_word' id='word_14_5' title='bbox 418 656 489 683'>37.45</span>
            </span>
            <span class='ocr_line' id='line_15' title='bbox 18 688 263 715; baseline 0 -5'>
                <span class='ocrx_word' id='word_15_0' title='bbox 18 688 76 714'>CARD</span>
                <span class='ocrx_word' id='word_15_1' title='bbox 88 689 115 713'>IS</span>
                <span class='ocrx_word' id='word_15_2' title='bbox 126 688 263 715'>REGISTERED</span>
            </span></div>
    </div>
  </body>
</html>
```



