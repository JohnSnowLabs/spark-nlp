---
layout: model
title: DiT model pretrained on IIT-CDIP and finetuned on RVL-CDIP for document classification
author: John Snow Labs
name: dit_base_finetuned_rvlcdip
date: 2022-06-09
tags: [en, licensed]
task: OCR Document Classification
language: en
edition: Visual NLP 4.0.0
spark_version: 3.2.1
supported: true
annotator: VisualDocumentClassifierv3
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

DiT was proposed in DiT: Self-supervised Pre-training for Document Image Transformer by Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei. DiT applies the self-supervised objective of BEiT (BERT pre-training of Image Transformers) to 42 million document images.  This model was trained for document image classification in the  RVL-CDIP dataset (a collection of 400,000 images belonging to one of 16 classes).

The abstract from the paper is the following: Image Transformer has recently achieved significant progress for natural image understanding, either using supervised (ViT, DeiT, etc.) or self-supervised (BEiT, MAE, etc.) pre-training techniques. In this paper, we propose DiT, a self-supervised pre-trained Document Image Transformer model using large-scale unlabeled text images for Document AI tasks, which is essential since no supervised counterparts ever exist due to the lack of human labeled document images. We leverage DiT as the backbone network in a variety of vision-based Document AI tasks, including document image classification, document layout analysis, as well as table detection. Experiment results have illustrated that the self-supervised pre-trained DiT model achieves new state-of-the-art results on these downstream tasks, e.g. document image classification (91.11 → 92.69), document layout analysis (91.0 → 94.9) and table detection (94.23 → 96.55).


## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/tutorials/Certification_Trainings/5.2.Visual_Document_Classifier_v3.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/dit_base_finetuned_rvlcdip_en_3.3.0_3.0_1654798502586.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from pyspark.ml import PipelineModel
from sparkocr.transformers import *

imagePath = "path to image"
bin_df = spark.read.format("binaryFile").load(imagePath).limit(50)
image_df = BinaryToImage().transform(bin_df)

binary_to_image = BinaryToImage()\
    .setOutputCol("image") \
    .setImageType(ImageType.TYPE_3BYTE_BGR)

doc_class = VisualDocumentClassifierV3() \
    .pretrained("dit_base_finetuned_rvlcdip", "en", "clinical/ocr") \
    .setInputCols(["image"]) \
    .setOutputCol("label")

# OCR pipeline
pipeline = PipelineModel(stages=[
    binary_to_image,
    doc_class
])

results = pipeline.transform(bin_df).cache()
```
```scala
import com.johnsnowlabs.ocr.transformers.*
import com.johnsnowlabs.ocr.OcrContext.implicits._

val imagePath = "path to image"
var imgDf = spark.read.format("binaryFile").load(imagePath)

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

## Example

### Input:
![Screenshot](../../_examples_ocr/image1.png)

### Output:
```bash
+-------+
|label  |
+-------+
|invoice|
+-------+
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dit_base_finetuned_rvlcdip|
|Type:|ocr|
|Compatibility:|Visual NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|319.6 MB|

## References

IIT-CDIP, RVL-CDIP