---
layout: model
title: RoBERTa Base Sentence Embeddings(sent_roberta_base)
author: John Snow Labs
name: sent_roberta_base
date: 2024-06-12
tags: [sentence_embeddings, en, english, roberta, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.4.0
spark_version: 3.4
supported: true
engine: onnx
annotator: RoBertaSentenceEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained model on English language using a masked language modeling (MLM) objective. It was introduced in this paper and first released in this repository. This model is case-sensitive: it makes a difference between english and English.

RoBERTa is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labeling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts.

More precisely, it was pretrained with the Masked language modeling (MLM) objective. Taking a sentence, the model randomly masks 15% of the words in the input then runs the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence.

This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences, for instance, you can train a standard classifier using the features produced by the BERT model as inputs.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_roberta_base_en_5.4.0_3.4_1718213024958.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_roberta_base_en_5.4.0_3.4_1718213024958.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = RoBertaSentenceEmbeddings.pretrained("sent_roberta_base", "en") \
      .setInputCols("sentence") \
      .setOutputCol("embeddings")
```
```scala
val embeddings = RoBertaSentenceEmbeddings.pretrained("sent_roberta_base", "en")
      .setInputCols("sentence")
      .setOutputCol("embeddings")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_roberta_base|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[sentence_embeddings]|
|Language:|en|
|Size:|297.8 MB|

## References

https://huggingface.co/FacebookAI/roberta-base