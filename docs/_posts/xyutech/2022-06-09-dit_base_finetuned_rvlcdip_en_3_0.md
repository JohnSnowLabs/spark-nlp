---
layout: model
title: DiT model pretrained on IIT-CDIP and finetuned on RVL-CDIP for document classification
author: John Snow Labs
name: dit_base_finetuned_rvlcdip
date: 2022-06-09
tags: [en, licensed]
task: OCR Document Classification
language: en
edition: Visual NLP 3.3.0
spark_version: 3.0
supported: true
annotator: VisualDocumentClassifierv3
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Document Image Transformer (DiT) model for document classification. The model was pretrained on IIT-CDIP dataset that includes 42 million document images and finetuned on RVL-CDIP dataset that consists of 400 000 grayscale images in 16 classes.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/dit_base_finetuned_rvlcdip_en_3.3.0_3.0_1654798502586.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/ocr/dit_base_finetuned_rvlcdip_en_3.3.0_3.0_1654798502586.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
        binary_to_image = BinaryToImage() \
            .setOutputCol("image") \
            .setImageType(ImageType.TYPE_3BYTE_BGR)

        doc_class = VisualDocumentClassifierV3() \
            .pretrained("dit_base_finetuned_rvlcdip", "en", "public/ocr/models") \
            .setInputCols(["image"]) \
            .setOutputCol("label")

        pipeline = PipelineModel(stages=[
            binary_to_image,
            doc_class,
        ])

```
```scala
    var bin2imTransformer = new BinaryToImage()
    bin2imTransformer.setImageType(ImageType.TYPE_3BYTE_BGR)

    val visualDocumentClassifier = VisualDocumentClassifierv3
        .pretrained("dit_base_finetuned_rvlcdip", "en", "public/ocr/models")
        .setInputCols(Array("image"))

    val pipeline = new Pipeline()
      .setStages(Array(
        bin2imTransformer,
        visualDocumentClassifier
      ))

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dit_base_finetuned_rvlcdip|
|Type:|ocr|
|Compatibility:|Visual NLP 3.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|319.6 MB|

## References

IIT-CDIP, RVL-CDIP