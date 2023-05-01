---
layout: model
title: Multilingual XLMRoBerta Embeddings (from jhu-clsp)
author: John Snow Labs
name: xlmroberta_embeddings_roberta_large_eng_ara_128k
date: 2022-05-13
tags: [en, ar, open_source, xlm_roberta, embeddings, xx]
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

Pretrained XLMRoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `roberta-large-eng-ara-128k` is a Multilingual model orginally trained by `jhu-clsp`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_embeddings_roberta_large_eng_ara_128k_xx_3.4.4_3.0_1652440561319.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_embeddings_roberta_large_eng_ara_128k_xx_3.4.4_3.0_1652440561319.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_roberta_large_eng_ara_128k","xx") \
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

val embeddings = XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_roberta_large_eng_ara_128k","xx") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("xx.embed.xlmr_roberta.large_128d").predict("""PUT YOUR STRING HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_embeddings_roberta_large_eng_ara_128k|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|xx|
|Size:|1.0 GB|
|Case sensitive:|true|

## References

- https://huggingface.co/jhu-clsp/roberta-large-eng-ara-128k