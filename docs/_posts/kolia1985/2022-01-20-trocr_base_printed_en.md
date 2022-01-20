---
layout: model
title: TrOCR
author: John Snow Labs
name: trocr_base_printed
date: 2022-01-20
tags: [en, licensed]
task: OCR Text Detection & Recognition
language: en
edition: Spark NLP 3.4.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The TrOCR model was proposed in TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei. TrOCR consists of an image Transformer encoder and an autoregressive text Transformer decoder to perform optical character recognition (OCR).

The abstract from the paper is the following:

Text recognition is a long-standing research problem for document digitalization. Existing approaches for text recognition are usually built based on CNN for image understanding and RNN for char-level text generation. In addition, another language model is usually needed to improve the overall accuracy as a post-processing step. In this paper, we propose an end-to-end text recognition approach with pre-trained image Transformer and text Transformer models, namely TrOCR, which leverages the Transformer architecture for both image understanding and wordpiece-level text generation. The TrOCR model is simple but effective, and can be pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. Experiments show that the TrOCR model outperforms the current state-of-the-art models on both printed and handwritten text recognition tasks.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/trocr_base_printed_en_3.4.0_2.4_1642690940455.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

ocr = TrOcr().pretrained("trocr_base_printed", "en", "clinical/ocr")
ocr.setInputCols(["image"])
ocr.setOutputCol("text")

result = ocr.transform(image_text_lines_df).collect()
print(result[0].text)

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
ocr = TrOcr().pretrained("trocr_base_printed", "en", "clinical/ocr")
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
|Model Name:|trocr_base_printed|
|Type:|ocr|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|779.8 MB|