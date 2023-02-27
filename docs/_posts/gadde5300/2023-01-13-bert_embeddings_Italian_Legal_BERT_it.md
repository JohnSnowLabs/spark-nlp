---
layout: model
title: Italian BERT Embedding  Cased model
author: John Snow Labs
name: bert_embeddings_Italian_Legal_BERT
date: 2023-01-13
tags: [it, open_source, embeddings, bert]
task: Embeddings
language: it
edition: Spark NLP 4.2.7
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BERT Embedding model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Italian-Legal-BERT` is a Italian model originally trained by `dlicari`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_Italian_Legal_BERT_it_4.2.7_3.0_1673598434160.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")
  
embeddings = BertEmbeddings.pretrained("bert_embeddings_Italian_Legal_BERT","it") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Adoro Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_Italian_Legal_BERT|
|Compatibility:|Spark NLP 4.2.7+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|it|
|Size:|411.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/dlicari/Italian-Legal-BERT
- https://colab.research.google.com/drive/1aXOmqr70fjm8lYgIoGJMZDsK0QRIL4Lt?usp=sharing