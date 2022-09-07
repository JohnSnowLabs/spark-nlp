---
layout: model
title: Tickets Binary Classification
author: John Snow Labs
name: finimgclf_vit_tickets
date: 2022-09-07
tags: [en, finance, classification, tickets, licensed]
task: Image Classification
language: en
edition: Spark NLP for Finance 1.0.0
spark_version: 3.2
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a ViT (Visual Transformer) model, which can be used to carry out Binary Classification (true or false) on pictures / photos / images. This model has been trained in-house with different corpora, including:
- CORD
- COCO
- In-house annotated tickets 

You can use this model to filter out non-tickets from a folder of images or mobile pictures, and then use Spark OCR to extract information using the layout and the text features.

## Predicted Entities

`ticket`, `no_ticket`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finimgclf_vit_tickets_en_1.0.0_3.2_1662559973697.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier_loaded = ViTForImageClassification.pretrained("finvisualclf_tickets", "en", "finance/models")\
  .setInputCols(["image_assembler"])\
  .setOutputCol("class")

pipeline = Pipeline().setStages([
    document_assembler,
    imageClassifier_loaded
])

test_image = spark.read\
    .format("image")\
    .option("dropInvalid", value = True)\
    .load("./ticket.JPEG")

result = pipeline.fit(test_image).transform(test_image)

result.select("class.result").show(1, False)
```

</div>

## Results

```bash
+--------+
|result  |
+--------+
|[ticket]|
+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finimgclf_vit_tickets|
|Compatibility:|Spark NLP for Finance 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|321.9 MB|

## References

Cord, rvl-cdip, visual-genome and an external receipt tickers dataset

## Benchmarking

```bash
training_loss validation_loss f1 
0.0006 0.0044 0.9997
```