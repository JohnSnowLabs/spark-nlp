---
layout: model
title: BioBERT Embeddings (Pubmed)
author: John Snow Labs
name: biobert_pubmed_base_cased_v1.2
date: 2023-06-21
tags: [bert, embeddings, en, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.0.0
spark_version: 3.4
supported: true
engine: onnx
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is the v1.2 of [biobert_pubmed_base_cased](https://nlp.johnsnowlabs.com/2020/09/19/biobert_pubmed_base_cased.html) model and contains pre-trained weights of BioBERT, a language representation model for biomedical domain, especially designed for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, question answering, etc. The details are described in the paper "[BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/abs/1901.08746v2)".

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biobert_pubmed_base_cased_v1.2_en_5.0.0_3.4_1687336480762.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/biobert_pubmed_base_cased_v1.2_en_5.0.0_3.4_1687336480762.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased_v1.2","en") \
      .setInputCols(["document", "token"]) \
      .setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I hate cancer"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
  .setInputCol("text") 
  .setOutputCol("document")

val tokenizer = new Tokenizer() 
  .setInputCols(Array("document"))
  .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased_v1.2","en") 
  .setInputCols(Array("document", "token")) 
  .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I hate cancer").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.biobert.pubmed.cased_base").predict("""I hate cancer""")
```

</div>

{:.model-param}

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

tokenizer = Tokenizer() \
      .setInputCols("document") \
      .setOutputCol("token")

embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased_v1.2","en") \
      .setInputCols(["document", "token"]) \
      .setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I hate cancer"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
  .setInputCol("text") 
  .setOutputCol("document")

val tokenizer = new Tokenizer() 
  .setInputCols(Array("document"))
  .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased_v1.2","en") 
  .setInputCols(Array("document", "token")) 
  .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I hate cancer").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("en.embed.biobert.pubmed.cased_base").predict("""I hate cancer""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|biobert_pubmed_base_cased_v1.2|
|Compatibility:|Spark NLP 5.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|403.6 MB|
|Case sensitive:|true|