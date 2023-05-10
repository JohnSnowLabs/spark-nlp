---
layout: model
title: Abstractive Summarization by BART - DistilBART XSUM
author: John Snow Labs
name: distilbart_xsum_12_6
date: 2023-05-09
tags: [bart, summarization, text_to_text, xsum, distil, en, open_source, tensorflow]
task: Summarization
language: en
edition: Spark NLP 4.4.2
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BartTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

"BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension Transformer" The Facebook BART (Bidirectional and Auto-Regressive Transformer) model is a state-of-the-art language generation model that was introduced by Facebook AI in 2019. It is based on the transformer architecture and is designed to handle a wide range of natural language processing tasks such as text generation, summarization, and machine translation.

This pre-trained model is DistilBART fine-tuned on the Extreme Summarization (XSum) Dataset.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbart_xsum_12_6_en_4.4.2_3.0_1683644681912.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbart_xsum_12_6_en_4.4.2_3.0_1683644681912.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|733.7 MB|

## References

https://huggingface.co/sshleifer/distilbart-xsum-12-6

## Benchmarking

```bash
### Metrics for DistilBART models

| Model Name                 |   MM Params |   Inference Time (MS) |   Speedup |   Rouge 2 |   Rouge-L |
|:---------------------------|------------:|----------------------:|----------:|----------:|----------:|
| distilbart-xsum-12-1       |         222 |                    90 |      2.54 |     18.31 |     33.37 |
| distilbart-xsum-6-6        |         230 |                   132 |      1.73 |     20.92 |     35.73 |
| distilbart-xsum-12-3       |         255 |                   106 |      2.16 |     21.37 |     36.39 |
| distilbart-xsum-9-6        |         268 |                   136 |      1.68 |     21.72 |     36.61 |
| bart-large-xsum (baseline) |         406 |                   229 |      1    |     21.85 |     36.50 |
| distilbart-xsum-12-6       |         306 |                   137 |      1.68 |     22.12 |     36.99 |
| bart-large-cnn (baseline)  |         406 |                   381 |      1    |     21.06 |     30.63 |
| distilbart-12-3-cnn        |         255 |                   214 |      1.78 |     20.57 |     30.00 |
| distilbart-12-6-cnn        |         306 |                   307 |      1.24 |     21.26 |     30.59 |
| distilbart-6-6-cnn         |         230 |                   182 |      2.09 |     20.17 |     29.70 |
```
