---
layout: model
title: Persian BertForMaskedLM Base Uncased model (from HooshvareLab)
author: John Snow Labs
name: bert_embeddings_fa_base_uncased
date: 2022-12-02
tags: [fa, open_source, bert_embeddings, bertformaskedlm]
task: Embeddings
language: fa
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-fa-base-uncased` is a Persian model originally trained by `HooshvareLab`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_fa_base_uncased_fa_4.2.4_3.0_1670019389838.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_fa_base_uncased","fa") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, bert_loaded])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols(Array("text")) 
    .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_fa_base_uncased","fa") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(True)    
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, bert_loaded))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_fa_base_uncased|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|fa|
|Size:|609.3 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/HooshvareLab/bert-fa-base-uncased
- https://github.com/hooshvare/parsbert
- https://arxiv.org/abs/2005.12515
- https://dumps.wikimedia.org/fawiki/
- https://github.com/miras-tech/MirasText
- https://bigbangpage.com/
- https://www.chetor.com/
- https://www.eligasht.com/Blog/
- https://www.digikala.com/mag/
- https://www.ted.com/talks
- https://github.com/hooshvare/parsbert/issues