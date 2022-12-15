---
layout: model
title: Chinese BERT Sentence Embeddings (from DMetaSoul)
author: John Snow Labs
name: sbert_chinese_qmc_finance_v1
date: 2022-12-07
tags: [zh, chinese, embeddings, transformer, open_source, tensorflow]
task: Embeddings
language: zh
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BERT Sentence Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `sbert-chinese-qmc-finance-v1` is a Chinese model originally trained by `DMetaSoul`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sbert_chinese_qmc_finance_v1_zh_4.2.4_3.0_1670419872566.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sentence_embeddings = BertSentenceEmbeddings.pretrained("sbert_chinese_qmc_finance_v1", "zh")\
  .setInputCols(["sentence"])\
  .setOutputCol("sbert_embeddings")
```
```scala
val sentence_embeddings = BertSentenceEmbeddings.pretrained("sbert_chinese_qmc_finance_v1", "zh")
  .setInputCols("sentence")
  .setOutputCol("bert_sentence"))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbert_chinese_qmc_finance_v1|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|zh|
|Size:|384.0 MB|
|Case sensitive:|true|

## References

https://huggingface.co/DMetaSoul/sbert-chinese-qmc-finance-v1