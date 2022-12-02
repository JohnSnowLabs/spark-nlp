---
layout: model
title: Korean Electra Embeddings (from snunlp)
author: John Snow Labs
name: electra_embeddings_kr_electra_generator
date: 2022-05-16
tags: [ko, open_source, electra, embeddings]
task: Embeddings
language: ko
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Electra Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `KR-ELECTRA-generator` is a Korean model orginally trained by `snunlp`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_embeddings_kr_electra_generator_ko_3.4.4_3.0_1652716594290.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
  
embeddings = BertEmbeddings.pretrained("electra_embeddings_kr_electra_generator","ko") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["나는 Spark NLP를 좋아합니다"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("electra_embeddings_kr_electra_generator","ko") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("나는 Spark NLP를 좋아합니다").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_embeddings_kr_electra_generator|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|ko|
|Size:|124.8 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/snunlp/KR-ELECTRA-generator
- https://github.com/google-research/electra
- https://github.com/google-research/electra
- https://bitbucket.org/eunjeon/mecab-ko-dic/src/master/
- https://drive.google.com/file/d/1L_yKEDaXM_yDLwHm5QrXAncQZiMN3BBU/view?usp=sharing
- https://github.com/monologg/KoELECTRA
- https://github.com/snunlp/KR-ELECTRA
- https://github.com/monologg/KoELECTRA