---
layout: model
title: RoBERTa large model
author: John Snow Labs
name: roberta_large
date: 2021-05-20
tags: [en, english, embeddings, roberta, open_source]
task: Embeddings
language: en
edition: Spark NLP 3.1.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained model on English language using a masked language modeling (MLM) objective. It was introduced in [this paper](https://arxiv.org/abs/1907.11692) and first released in [this repository](https://github.com/pytorch/fairseq/tree/master/examples/roberta). This model is case-sensitive: it makes a difference between english and English.

RoBERTa is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labeling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. 

More precisely, it was pretrained with the Masked language modeling (MLM) objective. Taking a sentence, the model randomly masks 15% of the words in the input then runs the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence.

This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences, for instance, you can train a standard classifier using the features produced by the BERT model as inputs.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_large_en_3.1.0_2.4_1621523610703.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = RoBertaEmbeddings.pretrained("roberta_large", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
```
```scala
val embeddings = RoBertaEmbeddings.pretrained("roberta_large", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.roberta.large").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_large|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/roberta-large](https://huggingface.co/roberta-large)

## Benchmarking

```bash
When fine-tuned on downstream tasks, this model achieves the following results:

Glue test results:

| Task | MNLI | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  |
|:----:|:----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|
|      | 90.2 | 92.2 | 94.7 | 96.4  | 68.0 | 96.4  | 90.9 | 86.6 |

```