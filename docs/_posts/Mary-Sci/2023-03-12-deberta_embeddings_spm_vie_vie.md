---
layout: model
title: Vietnamese Deberta Embeddings model (from hieule)
author: John Snow Labs
name: deberta_embeddings_spm_vie
date: 2023-03-12
tags: [deberta, open_source, deberta_embeddings, debertav2formaskedlm, vie, tensorflow]
task: Embeddings
language: vie
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: DeBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DebertaEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `spm-vie-deberta` is a Vietnamese model originally trained by `hieule`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_embeddings_spm_vie_vie_4.3.1_3.0_1678627522214.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_embeddings_spm_vie_vie_4.3.1_3.0_1678627522214.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = DeBertaEmbeddings.pretrained("deberta_embeddings_spm_vie","vie") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

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
 
val embeddings = DeBertaEmbeddings.pretrained("deberta_embeddings_spm_vie","vie") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(true)    
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_embeddings_spm_vie|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|vie|
|Size:|290.3 MB|
|Case sensitive:|false|

## References

https://huggingface.co/hieule/spm-vie-deberta