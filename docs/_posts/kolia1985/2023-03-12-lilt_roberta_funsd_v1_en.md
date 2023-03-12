---
layout: model
title: LiLT-RoBERTa fine-tuned on Funsd
author: John Snow Labs
name: lilt_roberta_funsd_v1
date: 2023-03-12
tags: [en, licensed]
task: Named Entity Recognition
language: en
edition: Visual NLP 4.3.1
spark_version: 3.0
supported: true
annotator: VisualDocumentNerLilt
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The Language-Independent Layout Transformer (LiLT) allows to combine any pre-trained RoBERTa encoder from the hub (hence, in any language) with a lightweight Layout Transformer to have a LayoutLM-like model for any language.

This model is a fine-tuned version of base model on the Funsd dataset. It achieves the following results on the evaluation set:

Loss: 1.6552
Precision: 0.8762
Recall: 0.8857
F1: 0.8809
Accuracy: 0.8068

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/lilt_roberta_funsd_v1_en_4.3.1_3.0_1678603416755.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/ocr/lilt_roberta_funsd_v1_en_4.3.1_3.0_1678603416755.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

visualDocumentNER = VisualDocumentNerLilt
      .pretrained("lilt_roberta_funsd_v1", "en", "clinical/ocr")
      .setInputCol("hocr")
      .setOutputCol("entities")

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
visualDocumentNER = VisualDocumentNerLilt
      .pretrained("lilt_roberta_funsd_v1", "en", "clinical/ocr")
      .setInputCol("hocr")
      .setOutputCol("entities")
```
```scala
val visualDocumentNER = VisualDocumentNerLilt
      .pretrained("lilt_roberta_funsd_v1", "en", "clinical/ocr")
      .setInputCol("hocr")
      .setOutputCol("entities")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|lilt_roberta_funsd_v1|
|Type:|ocr|
|Compatibility:|Visual NLP 4.3.1+|
|License:|Licensed|
|Edition:|Official|
|Output Labels:|[entities]|
|Language:|en|
|Size:|440.0 MB|

## Benchmarking

```bash
Loss: 1.6552
Precision: 0.8762
Recall: 0.8857
F1: 0.8809
Accuracy: 0.8068
```