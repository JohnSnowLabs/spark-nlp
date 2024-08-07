---
layout: model
title: Abstractive Summarization by BART - DistilBART XSUM
author: John Snow Labs
name: distilbart_xsum_12_6
date: 2024-08-07
tags: [en, summarization, text_to_text, distil, open_source, tensorflow]
task: Summarization
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BartTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

“BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension Transformer” The Facebook BART (Bidirectional and Auto-Regressive Transformer) model is a state-of-the-art language generation model that was introduced by Facebook AI in 2019. It is based on the transformer architecture and is designed to handle a wide range of natural language processing tasks such as text generation, summarization, and machine translation.

This pre-trained model is DistilBART fine-tuned on the Extreme Summarization (XSum) Dataset.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbart_xsum_12_6_en_5.4.2_3.0_1723052800054.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbart_xsum_12_6_en_5.4.2_3.0_1723052800054.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

bart = BartTransformer.pretrained("distilbart_xsum_12_6") \
            .setTask("summarize:") \
            .setMaxOutputLength(200) \
            .setInputCols(["documents"]) \
            .setOutputCol("summaries")

```
```scala

val bart = BartTransformer.pretrained("distilbart_xsum_12_6")
            .setTask("summarize:")
            .setMaxOutputLength(200)
            .setInputCols("documents")
            .setOutputCol("summaries")


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbart_xsum_12_6|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[summaries]|
|Language:|en|
|Size:|733.6 MB|