---
layout: model
title: Receipts Binary Classification
author: John Snow Labs
name: finvisualclf_vit_tickets
date: 2022-09-07
tags: [en, finance, classification, tickets, licensed]
task: Image Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
recommended: true
annotator: ViTForImageClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a ViT (Visual Transformer) model, which can be used to carry out Binary Classification (true or false) on pictures / photos / images. This model has been trained in-house with different corpora, including:
- CORD
- COCO
- In-house annotated receipts 

You can use this model to filter out non-tickets from a folder of images or mobile pictures, and then use Visual NLP to extract information using the layout and the text features.

## Predicted Entities

`ticket`, `no_ticket`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finvisualclf_vit_tickets_en_1.0.0_3.2_1662560058841.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finvisualclf_vit_tickets_en_1.0.0_3.2_1662560058841.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier_loaded = nlp.ViTForImageClassification.pretrained("finvisualclf_vit_tickets", "en", "finance/models")\
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
|Model Name:|finvisualclf_vit_tickets|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|321.9 MB|

## References

Cord, rvl-cdip, visual-genome and an external receipt dataset

## Benchmarking

```bash
label            score
training_loss    0.0006  
validation_loss  0.0044
f1               0.9997
```
