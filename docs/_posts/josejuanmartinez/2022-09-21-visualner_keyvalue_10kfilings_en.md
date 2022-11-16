---
layout: model
title: Key Value Recognition on 10K filings
author: John Snow Labs
name: visualner_keyvalue_10kfilings
date: 2022-09-21
tags: [en, licensed]
task: OCR Object Detection
language: en
edition: Visual NLP 4.0.0
spark_version: 3.2
supported: true
annotator: VisualDocumentNERv21
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Form Recognition / Key Value extraction model, trained on the summary page of SEC 10K filings. It extracts KEY, VALUE or HEADER as entities, being HEADER the title on the filing.

## Predicted Entities

`KEY`, `VALUE`, `HEADER`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/visualner_keyvalue_10kfilings_en_4.0.0_3.2_1663781115795.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
binary_to_image = BinaryToImage()\
    .setOutputCol("image") \
    .setImageType(ImageType.TYPE_3BYTE_BGR)

img_to_hocr = ImageToHocr()\
    .setInputCol("image")\
    .setOutputCol("hocr")\
    .setIgnoreResolution(False)\
    .setOcrParams(["preserve_interword_spaces=0"])

tokenizer = HocrTokenizer()\
    .setInputCol("hocr")\
    .setOutputCol("token")

doc_ner = VisualDocumentNerV21()\
    .pretrained("visualner_keyvalue_10kfilings", "en", "clinical/ocr")\
    .setInputCols(["token", "image"])\
    .setOutputCol("entities")

draw = ImageDrawAnnotations() \
    .setInputCol("image") \
    .setInputChunksCol("entities") \
    .setOutputCol("image_with_annotations") \
    .setFontSize(10) \
    .setLineWidth(4)\
    .setRectColor(Color.red)

# OCR pipeline
pipeline = PipelineModel(stages=[
    binary_to_image,
    img_to_hocr,
    tokenizer,
    doc_ner,
    draw
])

import pkg_resources
bin_df = spark.read.format("binaryFile").load('data/t01.jpg')
bin_df.show()

results = pipeline.transform(bin_df).cache()

res = results.collect()

## since pyspark2.3 doesn't have element_at, 'getItem' is involked
path_array = f.split(results['path'], '/')

# from pyspark2.4
# results.withColumn("filename", f.element_at(f.split("path", "/"), -1)) \

results.withColumn('filename', path_array.getItem(f.size(path_array)- 1)) \
    .withColumn("exploded_entities", f.explode("entities")) \
    .select("filename", "exploded_entities") \
    .show(truncate=False)
            
```

</div>

## Results

```bash
+--------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|filename|exploded_entities                                                                                                                                        |
+--------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|t01.jpg |{named_entity, 268, 269, OTHERS, {confidence -> 96, width -> 14, x -> 822, y -> 1101, word -> of, token -> of, height -> 34}, []}                        |
|t01.jpg |{named_entity, 271, 273, OTHERS, {confidence -> 89, width -> 33, x -> 837, y -> 1112, word -> the, token -> the, height -> 13}, []}                      |
|t01.jpg |{named_entity, 275, 277, OTHERS, {confidence -> 89, width -> 30, x -> 874, y -> 1113, word -> Act., token -> act, height -> 12}, []}                     |
|t01.jpg |{named_entity, 280, 282, KEY-B, {confidence -> 94, width -> 26, x -> 910, y -> 1113, word -> Yes, token -> yes, height -> 12}, []}                       |
|t01.jpg |{named_entity, 284, 285, VALUE-B, {confidence -> 45, width -> 13, x -> 944, y -> 1112, word -> LI, token -> li, height -> 13}, []}                       |
|t01.jpg |{named_entity, 287, 288, KEY-B, {confidence -> 83, width -> 22, x -> 963, y -> 1113, word -> No, token -> no, height -> 12}, []}                         |
|t01.jpg |{named_entity, 290, 295, HEADER-B, {confidence -> 96, width -> 89, x -> 1493, y -> 13, word -> UNITED, token -> united, height -> 16}, []}               |
|t01.jpg |{named_entity, 297, 302, HEADER-I, {confidence -> 95, width -> 83, x -> 1590, y -> 13, word -> STATES, token -> states, height -> 16}, []}               |
|t01.jpg |{named_entity, 304, 313, HEADER-B, {confidence -> 95, width -> 221, x -> 1186, y -> 45, word -> SECURITIES, token -> securities, height -> 25}, []}      |
|t01.jpg |{named_entity, 315, 317, HEADER-I, {confidence -> 95, width -> 80, x -> 1415, y -> 45, word -> AND, token -> and, height -> 25}, []}                     |
|t01.jpg |{named_entity, 319, 326, HEADER-I, {confidence -> 96, width -> 212, x -> 1507, y -> 45, word -> EXCHANGE, token -> exchange, height -> 25}, []}          |
|t01.jpg |{named_entity, 328, 337, HEADER-I, {confidence -> 95, width -> 249, x -> 1732, y -> 45, word -> COMMISSION, token -> commission, height -> 25}, []}      |
|t01.jpg |{named_entity, 339, 348, HEADER-B, {confidence -> 96, width -> 125, x -> 1461, y -> 86, word -> Washington,, token -> washington, height -> 21}, []}     |
|t01.jpg |{named_entity, 351, 351, HEADER-I, {confidence -> 93, width -> 43, x -> 1595, y -> 86, word -> D.C., token -> d, height -> 16}, []}                      |
|t01.jpg |{named_entity, 356, 360, HEADER-I, {confidence -> 93, width -> 59, x -> 1646, y -> 86, word -> 20549, token -> 20549, height -> 16}, []}                 |
|t01.jpg |{named_entity, 362, 365, HEADER-B, {confidence -> 93, width -> 112, x -> 1484, y -> 159, word -> FORM, token -> form, height -> 25}, []}                 |
|t01.jpg |{named_entity, 367, 368, HEADER-I, {confidence -> 91, width -> 77, x -> 1609, y -> 159, word -> 10-K, token -> 10, height -> 25}, []}                    |
+--------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|visualner_keyvalue_10kfilings|
|Type:|ocr|
|Compatibility:|Visual NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|744.3 MB|

## References

Sec 10K filings