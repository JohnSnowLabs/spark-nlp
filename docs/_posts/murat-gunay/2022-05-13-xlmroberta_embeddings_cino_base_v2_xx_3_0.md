---
layout: model
title: Multilingual XLMRoBerta Embeddings (from hfl)
author: John Snow Labs
name: xlmroberta_embeddings_cino_base_v2
date: 2022-05-13
tags: [zh, ko, open_source, xlm_roberta, embeddings, xx, cino]
task: Embeddings
language: xx
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: XlmRoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XLMRoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `cino-base-v2` is a Multilingual model orginally trained by `hfl`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_embeddings_cino_base_v2_xx_3.4.4_3.0_1652439334973.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_embeddings_cino_base_v2_xx_3.4.4_3.0_1652439334973.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_cino_base_v2","xx") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_cino_base_v2","xx") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_embeddings_cino_base_v2|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|xx|
|Size:|712.7 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/hfl/cino-base-v2
- https://github.com/ymcui/Chinese-Minority-PLM
- https://github.com/ymcui/MacBERT
- https://github.com/ymcui/Chinese-BERT-wwm
- https://github.com/ymcui/Chinese-ELECTRA
- https://github.com/ymcui/Chinese-XLNet
- https://github.com/airaria/TextBrewer
- https://github.com/ymcui/HFL-Anthology
