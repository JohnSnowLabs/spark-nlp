---
layout: model
title: Chinese BERT Base
author: John Snow Labs
name: bert_base_chinese
date: 2021-05-20
tags: [zh, chinese, bert, embeddings, open_source]
task: Embeddings
language: zh
edition: Spark NLP 3.1.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture. It was originally published by

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2018.

The weights of this model are those released by the original BERT authors. This model has been pre-trained for Chinese on Wikipedia. For training, random input masking has been applied independently to word pieces (as in the original BERT paper).

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_chinese_zh_3.1.0_2.4_1621517505756.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_chinese_zh_3.1.0_2.4_1621517505756.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_base_chinese", "zh") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])

```
```scala
val embeddings = BertEmbeddings.pretrained("bert_base_chinese", "zh")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))

```


{:.nlu-block}
```python
import nlu
nlu.load("zh.embed").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_chinese|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|zh|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/bert-base-chinese](https://huggingface.co/bert-base-chinese)