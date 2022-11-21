---
layout: model
title: Visual NER - CORD (Receipts)
author: John Snow Labs
name: visualner_receipts
date: 2022-09-21
tags: [xx, licensed]
task: OCR Object Detection
language: xx
edition: Visual NLP 4.0.0
spark_version: 3.2
supported: true
annotator: VisualDocumentNERv21
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Visual NER, a model trained on the top of LayoutLMV2 to detect regions in Tickets. This model can be used after, for example, the Binary Image Classifier of Tickets, available at https://nlp.johnsnowlabs.com/2022/09/07/finvisualclf_vit_tickets_en.html

## Predicted Entities

`COMPANY`, `DATE`, `AMOUNT`, `NAME`, `NUM`, `UNITPRICE`, `CNT`, `DISCOUNTPRICE`, `PRICE`, `ITEMSUBTOTAL`, `VATyn`, `SUBTOTAL`, `TOTALDISCOUNT`, `SERVICEPRICE`, `OTHERSVCPRICE`, `TAX`, `TOTAL`, `CASH`, `CHANGE`, `CREDITCARD`, `EMONEY`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/visualner_receipts_xx_4.0.0_3.2_1663753935456.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    .pretrained("visualner_receipts", "en", "clinical/ocr")\
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
+----------+-------------------------------------------------------------------------------------------------------------------------------------------+
|filename  |exploded_entities                                                                                                                          |
+----------+-------------------------------------------------------------------------------------------------------------------------------------------+
|test0.jpeg|{named_entity, 24, 24, UNITPRICE-B, {confidence -> 95, width -> 66, x -> 306, y -> 229, word -> #010029, token -> #, height -> 17}, []}    |
|test0.jpeg|{named_entity, 32, 35, NAME-B, {confidence -> 91, width -> 38, x -> 200, y -> 250, word -> Sale, token -> sale, height -> 17}, []}         |
|test0.jpeg|{named_entity, 37, 37, OTHERS, {confidence -> 91, width -> 8, x -> 249, y -> 253, word -> #, token -> #, height -> 15}, []}                |
|test0.jpeg|{named_entity, 39, 47, NUM-B, {confidence -> 96, width -> 83, x -> 270, y -> 252, word -> 143710882, token -> 143710882, height -> 17}, []}|
|test0.jpeg|{named_entity, 49, 52, NAME-B, {confidence -> 96, width -> 37, x -> 191, y -> 274, word -> Team, token -> team, height -> 17}, []}         |
|test0.jpeg|{named_entity, 66, 68, CNT-B, {confidence -> 88, width -> 28, x -> 82, y -> 296, word -> Jan, token -> jan, height -> 16}, []}             |
|test0.jpeg|{named_entity, 114, 114, OTHERS, {confidence -> 63, width -> 27, x -> 229, y -> 323, word -> ***, token -> *, height -> 13}, []}           |
+----------+-------------------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|visualner_receipts|
|Type:|ocr|
|Compatibility:|Visual NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|xx|
|Size:|744.4 MB|

## References

CORD
