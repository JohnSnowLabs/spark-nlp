---
layout: model
title: Multilingual XLMRoBerta Embeddings Cased Model
author: John Snow Labs
name: xlmroberta_embeddings_paraphrase_mpnet_base_v2
date: 2023-06-29
tags: [xx, embeddings, xlmroberta, open_source, transformer, tensorflow]
task: Embeddings
language: xx
edition: Spark NLP 4.4.4
spark_version: 3.0
supported: true
engine: tensorflow
annotator: XlmRoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XLMRoberta Embeddings model is a multilingual embedding model adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_embeddings_paraphrase_mpnet_base_v2_xx_4.4.4_3.0_1688073546075.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_embeddings_paraphrase_mpnet_base_v2_xx_4.4.4_3.0_1688073546075.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_paraphrase_mpnet_base_v2","xx") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, 
                            tokenizer, 
                            embeddings])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")
result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols("document")
    .setOutputCol("token")

val embeddings = XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_paraphrase_mpnet_base_v2", "xx") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, 
                                              tokenizer, 
                                              embeddings))

val data = Seq("I love Spark NLP").toDS.toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_embeddings_paraphrase_mpnet_base_v2|
|Compatibility:|Spark NLP 4.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|xx|
|Size:|1.0 GB|
|Case sensitive:|true|

## References

https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2