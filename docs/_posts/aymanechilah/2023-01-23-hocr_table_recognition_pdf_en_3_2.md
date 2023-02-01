---
layout: model
title: Hocr for table recognition pdf
author: John Snow Labs
name: hocr_table_recognition_pdf
date: 2023-01-23
tags: [en, licensed]
task: HOCR Table Recognition
language: en
edition: Visual NLP 4.2.4
spark_version: 3.2.1
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Table structure recognition based on hocr with Tesseract architecture, for PDF documents. 

Tesseract has been trained on a variety of datasets to improve its recognition capabilities. These datasets include images of text in various languages and scripts, as well as images with different font styles, sizes, and orientations. The training process involves feeding the engine with a large number of images and their corresponding text, allowing the engine to learn the patterns and characteristics of different text styles. One of the most important datasets used in training Tesseract is the UNLV dataset, which contains over 400,000 images of text in different languages, scripts, and font styles. This dataset is widely used in the OCR community and has been instrumental in improving the accuracy of Tesseract. Other datasets that have been used in training Tesseract include the ICDAR dataset, the IIIT-HWS dataset, and the RRC-GV-WS dataset.

In addition to these datasets, Tesseract also uses a technique called adaptive training, where the engine can continuously improve its recognition capabilities by learning from new images and text. This allows Tesseract to adapt to new text styles and languages, and improve its overall accuracy.


## Predicted Entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-ocr-workshop/tree/master/jupyter/Cards/SparkOCRPdfToTable.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
<button class="button button-orange" disabled>Download</button>


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
pdf_to_hocr = PdfToHocr() \
        .setInputCol("content") \
        .setOutputCol("hocr")

tokenizer = HocrTokenizer() \
        .setInputCol("hocr") \
        .setOutputCol("token") \

pdf_to_image = PdfToImage() \
        .setInputCol("content") \
        .setOutputCol("image") \
        .setPageNumCol("tmp_pagenum") \
        .setImageType(ImageType.TYPE_3BYTE_BGR)

table_detector = ImageTableDetector \
        .pretrained("general_model_table_detection_v2", "en", "public/ocr/models") \
        .setInputCol("image") \
        .setOutputCol("table_regions") \
        .setScoreThreshold(0.9) \
        .setApplyCorrection(True) \
        .setScaleWidthToCol("width_dimension") \
        .setScaleHeightToCol("height_dimension")

image_scaler = ImageScaler() \
        .setWidthCol("width_dimension") \
        .setHeightCol("height_dimension")

hocr_to_table = HocrToTextTable() \
        .setInputCol("hocr") \
        .setRegionCol("table_regions") \
        .setOutputCol("tables")

draw_annotations = ImageDrawAnnotations() \
        .setInputCol("scaled_image") \
        .setInputChunksCol("tables") \
        .setOutputCol("image_with_annotations") \
        .setFilledRect(False) \
        .setFontSize(5) \
        .setRectColor(Color.red)

draw_regions = ImageDrawRegions() \
        .setInputCol("scaled_image") \
        .setInputRegionsCol("table_regions") \
        .setOutputCol("image_with_regions") \
        .setRectColor(Color.red)

pipeline1 = PipelineModel(stages=[
        pdf_to_hocr,
        tokenizer,
        pdf_to_image,
        table_detector,
        image_scaler,
        draw_regions,
        hocr_to_table
])

test_image_path = "data/pdfs/f1120.pdf"
bin_df = spark.read.format("binaryFile").load(test_image_path)

result = pipeline1.transform(bin_df).cache().drop("tmp_pagenum")
result = result.filter(result.pagenum == 1)
```
```scala
val pdf_to_hocr = new PdfToHocr() 
        .setInputCol("content") 
        .setOutputCol("hocr")

val tokenizer = new HocrTokenizer() 
        .setInputCol("hocr") 
        .setOutputCol("token") 

val pdf_to_image = new PdfToImage() 
        .setInputCol("content") 
        .setOutputCol("image") 
        .setPageNumCol("tmp_pagenum") 
        .setImageType(ImageType.TYPE_3BYTE_BGR)

val table_detector = ImageTableDetector 
        .pretrained("general_model_table_detection_v2", "en", "public/ocr/models") 
        .setInputCol("image") 
        .setOutputCol("table_regions") 
        .setScoreThreshold(0.9) 
        .setApplyCorrection(True) 
        .setScaleWidthToCol("width_dimension") 
        .setScaleHeightToCol("height_dimension")

val image_scaler = new ImageScaler() 
        .setWidthCol("width_dimension") 
        .setHeightCol("height_dimension")

val hocr_to_table = new HocrToTextTable() 
        .setInputCol("hocr") 
        .setRegionCol("table_regions") 
        .setOutputCol("tables")

val draw_annotations = new ImageDrawAnnotations() 
        .setInputCol("scaled_image") 
        .setInputChunksCol("tables") 
        .setOutputCol("image_with_annotations") 
        .setFilledRect(False) 
        .setFontSize(5) 
        .setRectColor(Color.red)

val draw_regions = new ImageDrawRegions() 
        .setInputCol("scaled_image") 
        .setInputRegionsCol("table_regions") 
        .setOutputCol("image_with_regions") 
        .setRectColor(Color.red)

val pipeline1 = new PipelineModel().setStages(Array(
         pdf_to_hocr, 
         tokenizer, 
         pdf_to_image, 
         table_detector, 
         image_scaler, 
         draw_regions, 
         hocr_to_table))
        
val test_image_path = "data/pdfs/f1120.pdf"
val bin_df = spark.read.format("binaryFile").load(test_image_path)

val result = pipeline1.transform(bin_df).cache().drop("tmp_pagenum")
result = result.filter(result.pagenum == 1)
```
</div>

## Example

{%- capture input_image -%}
![Screenshot](/assets/images/examples_ocr/image14.png)
{%- endcapture -%}

{%- capture output_image -%}
![Screenshot](/assets/images/examples_ocr/image14_out.png)
{%- endcapture -%}


{% include templates/input_output_image.md
input_image=input_image
output_image=output_image
%}

## Output text

```bash
path	modificationTime	length	hocr	height_dimension	width_dimension	pagenum	token	image	total_pages	documentnum	table_regions	scaled_image	image_with_regions	tables	exception	table_index
file:/content/f11...	2023-01-23 08:16:...	3471478	<div title="bbox ...	791	611	1	[{token, 0, 2, 23...	{file:/content/f1...	1	0	{0, 0, 32.839153,...	{file:/content/f1...	{file:/content/f1...	{0, 0, 0.0, 0.0,...	null	0
```

```bash
Filename: f1120.pdf
Page:     1
Table:    0
5
col0	col1	col2	col3	col4
0	Schedule C	Dividends instructions , ) Inclusions , and Sp...	( a ) Dividends inclusions and	( b ) %	( c ) Special ( a ) × deductions ( b )
1	1	Dividends from less - than - 20 % - owned dome...	234	50	None
2	2	Dividends from 20 % - or - more - owned domest...	324123	65	None
3	3	Dividends on certain debt - financed stock of ...	324	instructions see	None
4	4	Dividends on certain preferred stock of less -...	234	23 . 3	None
5	5	Dividends on certain preferred stock of 20 % -...	42134	26 . 7	None
6	6	Dividends from less - than - 20 % - owned fore...	4234	50	None
7	7	Dividends from 20 % - or - more - owned foreig...	4234	65	None
8	8	Dividends from wholly owned foreign subsidiaries	42348987	100	None
9	9	Subtotal . Add lines 1 through 8 . See instruc...	987	instructions see	None
10	10	Dividends from domestic corporations received ...	9786	100	None
11	11	Dividends from affiliated group members .	789	100	None
12	12	Dividends from certain FSCs	0.00	100	None
13	13	Foreign - source portion of dividends received...	421.34	100	None
14	14	Dividends from foreign corporations not includ...	2341.23	None	None
15	15	Section 965 ( a ) inclusion .	1234.14	instructions see	None
16	16a	Subpart F inclusions derived from the sale by ...	46.54	100	None
17	b	Subpart F inclusions derived from hybrid divid...	6453.65	None	None
18	c	Other ( attach inclusions Form ( s ) 5471 from...	985.76	None	None
19	17	Global Intangible Low - Taxed Income ( GILTI )...	23.41	None	None
20	18	Gross - up for foreign taxes deemed paid	1.01	None	None
21	19	IC - DISC and former DISC dividends not includ...	3123.91	None	None
22	20	Other dividends	12.23	None	None
23	21	Deduction for dividends paid on certain prefer...			1.23
24	22	Section 250 deduction ( attach Form 8993 )			3.41
25	23	Total dividends and inclusions . Add column ( ...	2341.23	None	None
```