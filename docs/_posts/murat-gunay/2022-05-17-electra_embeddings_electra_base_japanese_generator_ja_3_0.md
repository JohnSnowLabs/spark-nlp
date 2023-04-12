---
layout: model
title: Japanese Electra Embeddings (from izumi-lab)
author: John Snow Labs
name: electra_embeddings_electra_base_japanese_generator
date: 2022-05-17
tags: [ja, open_source, electra, embeddings]
task: Embeddings
language: ja
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Electra Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `electra-base-japanese-generator` is a Japanese model orginally trained by `izumi-lab`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_embeddings_electra_base_japanese_generator_ja_3.4.4_3.0_1652786593799.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/electra_embeddings_electra_base_japanese_generator_ja_3.4.4_3.0_1652786593799.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = BertEmbeddings.pretrained("electra_embeddings_electra_base_japanese_generator","ja") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Spark NLPが大好きです"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("electra_embeddings_electra_base_japanese_generator","ja") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Spark NLPが大好きです").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_embeddings_electra_base_japanese_generator|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|ja|
|Size:|132.8 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/izumi-lab/electra-base-japanese-generator
- https://github.com/google-research/electra
- https://github.com/retarfi/language-pretraining/tree/v1.0
- https://github.com/google-research/electra
- https://arxiv.org/abs/2003.10555
- https://creativecommons.org/licenses/by-sa/4.0/