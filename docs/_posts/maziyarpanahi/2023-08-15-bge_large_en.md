---
layout: model
title: BAAI general embedding English (bge_large)
author: John Snow Labs
name: bge_large
date: 2023-08-15
tags: [open_source, bert, embeddings, english, en, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.0.2
spark_version: 3.0
supported: true
engine: onnx
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

FlagEmbedding can map any text to a low-dimensional dense vector which can be used for tasks like retrieval, classification,  clustering, or semantic search.
And it also can be used in vector database for LLMs.

`bge` is short for `BAAI general embedding`.

|              Model              | Language | Description | query instruction for retrieval\* |
|:-------------------------------|:--------:| :--------:| :--------:|
|  [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) |   English |  rank **1st** in [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en) |   English |  rank **2nd** in [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en) |   English | a small-scale model but with competitive performance  | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) |   Chinese | rank **1st** in [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) benchmark | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) |   Chinese | This model is trained without instruction, and rank **2nd** in [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) benchmark |   |
|  [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) |   Chinese |  a base-scale model but has similar ability with `bge-large-zh` | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) |   Chinese | a small-scale model but with competitive performance | `为这个句子生成表示以用于检索相关文章：`  |

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bge_large_en_5.0.2_3.0_1692109963281.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bge_large_en_5.0.2_3.0_1692109963281.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
                
document = DocumentAssembler()\ 
    .setInputCol("text")\ 
    .setOutputCol("document")

tokenizer = Tokenizer()\ 
    .setInputCols(["document"])\ 
    .setOutputCol("token") 

embeddings = BertEmbeddings.pretrained("bge_large", "en")\ 
    .setInputCols(["document", "token"])\ 
    .setOutputCol("embeddings")

```
```scala

val document = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols("document") 
    .setOutputCol("token")
    
val embeddings = BertEmbeddings.pretrained("bge_large", "en")
    .setInputCols("document", "token")
    .setOutputCol("embeddings")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bge_large|
|Compatibility:|Spark NLP 5.0.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|794.2 MB|
|Case sensitive:|true|

## References

BAAI models are from [BAAI](https://huggingface.co/BAAI)