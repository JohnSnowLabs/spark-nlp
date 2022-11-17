---
layout: model
title: LayoutLMv3 finetuned on funsd
author: John Snow Labs
name: layoutlmv3_finetuned_funsd
date: 2022-07-16
tags: [en, licensed]
task: Named Entity Recognition
language: en
edition: Visual NLP 3.14.0
spark_version: 3.2
supported: true
annotator: VisualDocumentNERv21
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This NER model is based on LayoutLMV3 pre-trained model and fine-tuned with FUNSD dataset

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/layoutlmv3_finetuned_funsd_en_3.14.0_3.2_1657982560895.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
ocr = ImageToHocr()\
            .setInputCol("image")\
            .setOutputCol("hocr")\
            .setIgnoreResolution(False)\
            .setOcrParams(["preserve_interword_spaces=0"])
        doc_ner = VisualDocumentNerV21()\
            .pretrained("layoutlmv3_finetuned_funsd", "en", "public/ocr/models") \
            .setInputCol("hocr")\
            .setOutputCol("entity")

```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|layoutlmv3_finetuned_funsd|
|Type:|ocr|
|Compatibility:|Visual NLP 3.14.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|467.0 MB|