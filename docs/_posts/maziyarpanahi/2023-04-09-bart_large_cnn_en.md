---
layout: model
title: BART (large-sized model), fine-tuned on CNN Daily Mail
author: John Snow Labs
name: bart_large_cnn
date: 2023-04-09
tags: [bart, summarization, cnn, text_to_text, en, english, open_source, tensorflow]
task: Summarization
language: en
edition: Spark NLP 4.4.0
spark_version: [3.2, 3.0]
supported: true
engine: tensorflow
annotator: BartTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

### BART (large-sized model), fine-tuned on CNN Daily Mail 

BART model pre-trained on English language, and fine-tuned on [CNN Daily Mail](https://huggingface.co/datasets/cnn_dailymail). It was introduced in the paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) by Lewis et al. and first released in [this repository (https://github.com/pytorch/fairseq/tree/master/examples/bart). 

Disclaimer: The team releasing BART did not write a model card for this model so this model card has been written by the Hugging Face team.

### Model description

BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder. BART is pre-trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text.

BART is particularly effective when fine-tuned for text generation (e.g. summarization, translation) but also works well for comprehension tasks (e.g. text classification, question answering). This particular checkpoint has been fine-tuned on CNN Daily Mail, a large collection of text-summary pairs.

### Intended uses & limitations

You can use this model for text summarization.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bart_large_cnn_en_4.4.0_3.2_1681068423111.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bart_large_cnn_en_4.4.0_3.2_1681068423111.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
bart = BartTransformer.pretrained("bart_large_cnn") \
            .setTask("summarize:") \
            .setMaxOutputLength(200) \
            .setInputCols(["documents"]) \
            .setOutputCol("summaries")
```
```scala
val bart = BartTransformer.pretrained("bart_large_cnn")
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
|Model Name:|bart_large_cnn|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[summaries]|
|Language:|en|
|Size:|1.1 GB|

## References

https://huggingface.co/datasets/cnn_dailymail