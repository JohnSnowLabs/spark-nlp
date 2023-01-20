---
layout: model
title: Visual NER on 10K Filings (SEC)
author: John Snow Labs
name: visualner_10kfilings
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

This model is a Visual NER team aimed to extract the main key points in the summary page of SEC 10 K filings (Annual reports).

## Predicted Entities

`REGISTRANT`, `ADDRESS`, `PHONE`, `DATE`, `EMPLOYERIDNB`, `EXCHANGE`, `STATE`, `STOCKCLASS`, `STOCKVALUE`, `TRADINGSYMBOL`, `FILENUMBER`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/visualner_10kfilings_en_4.0.0_3.2_1663769328577.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/ocr/visualner_10kfilings_en_4.0.0_3.2_1663769328577.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    .pretrained("visualner_10kfilings", "en", "clinical/models")\
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
+--------+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|filename|exploded_entities                                                                                                                                         |
+--------+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|t01.jpg |{named_entity, 712, 716, OTHERS, {confidence -> 96, width -> 74, x -> 1557, y -> 416, word -> Ended, token -> ended, height -> 18}, []}                   |
|t01.jpg |{named_entity, 718, 724, DATE-B, {confidence -> 96, width -> 97, x -> 1639, y -> 416, word -> January, token -> january, height -> 24}, []}               |
|t01.jpg |{named_entity, 726, 727, DATE-I, {confidence -> 95, width -> 34, x -> 1743, y -> 416, word -> 31,, token -> 31, height -> 22}, []}                        |
|t01.jpg |{named_entity, 730, 733, DATE-I, {confidence -> 96, width -> 54, x -> 1785, y -> 416, word -> 2021, token -> 2021, height -> 18}, []}                     |
|t01.jpg |{named_entity, 735, 744, OTHERS, {confidence -> 91, width -> 143, x -> 1372, y -> 472, word -> Commission, token -> commission, height -> 18}, []}        |
|t01.jpg |{named_entity, 746, 749, OTHERS, {confidence -> 96, width -> 36, x -> 1523, y -> 472, word -> file, token -> file, height -> 18}, []}                     |
|t01.jpg |{named_entity, 751, 756, OTHERS, {confidence -> 92, width -> 96, x -> 1568, y -> 472, word -> number:, token -> number, height -> 18}, []}                |
|t01.jpg |{named_entity, 759, 761, FILENUMBER-B, {confidence -> 92, width -> 119, x -> 1675, y -> 472, word -> 001-39495, token -> 001, height -> 18}, []}          |
|t01.jpg |{named_entity, 769, 773, REGISTRANT-B, {confidence -> 92, width -> 136, x -> 1472, y -> 558, word -> ASANA,, token -> asana, height -> 31}, []}           |
|t01.jpg |{named_entity, 776, 778, REGISTRANT-I, {confidence -> 95, width -> 72, x -> 1620, y -> 558, word -> INC., token -> inc, height -> 25}, []}        
+--------+----------------------------------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|visualner_10kfilings|
|Type:|ocr|
|Compatibility:|Visual NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|744.4 MB|

## References

SEC 10k filings
