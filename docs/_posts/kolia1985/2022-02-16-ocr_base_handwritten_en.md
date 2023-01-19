---
layout: model
title: Оcr base for handwritten text
author: John Snow Labs
name: ocr_base_handwritten
date: 2022-02-16
tags: [en, licensed]
task: OCR Text Detection & Recognition
language: en
edition: Visual NLP 3.3.3
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Ocr base model for recognise handwritten text based on TrOcr architecture.  The TrOCR model was proposed in TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei. TrOCR consists of an image Transformer encoder and an autoregressive text Transformer decoder to perform optical character recognition (OCR).  The abstract from the paper is the following:  Text recognition is a long-standing research problem for document digitalization. Existing approaches for text recognition are usually built based on CNN for image understanding and RNN for char-level text generation. In addition, another language model is usually needed to improve the overall accuracy as a post-processing step. In this paper, we propose an end-to-end text recognition approach with pre-trained image Transformer and text Transformer models, namely TrOCR, which leverages the Transformer architecture for both image understanding and wordpiece-level text generation. The TrOCR model is simple but effective, and can be pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. Experiments show that the TrOCR model outperforms the current state-of-the-art models on both printed and handwritten text recognition tasks.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/ocr_base_handwritten_en_3.3.3_2.4_1645034046021.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/ocr/ocr_base_handwritten_en_3.3.3_2.4_1645034046021.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

ocr = ImageToTextv2().pretrained("ocr_base_handwritten", "en", "clinical/ocr")
ocr.setInputCols(["image"])
ocr.setOutputCol("text")

result = ocr.transform(image_text_lines_df).collect()
print(result[0].text)

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
ocr = ImageToTextv2().pretrained("ocr_base_handwritten", "en", "clinical/ocr")
ocr.setInputCols(["image"])
ocr.setOutputCol("text")

result = ocr.transform(image_text_lines_df).collect()
print(result[0].text)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ocr_base_handwritten|
|Type:|ocr|
|Compatibility:|Visual NLP 3.3.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|781.9 MB|