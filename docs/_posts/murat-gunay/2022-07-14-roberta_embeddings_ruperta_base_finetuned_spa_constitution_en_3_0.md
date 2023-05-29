---
layout: model
title: English RoBERTa Embeddings Base Cased model (from mrm8488)
author: John Snow Labs
name: roberta_embeddings_ruperta_base_finetuned_spa_constitution
date: 2022-07-14
tags: [en, open_source, roberta, embeddings]
task: Embeddings
language: en
nav_key: models
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: RoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `RuPERTa-base-finetuned-spa-constitution` is a English model originally trained by `mrm8488`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_ruperta_base_finetuned_spa_constitution_en_4.0.0_3.0_1657809796803.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_embeddings_ruperta_base_finetuned_spa_constitution_en_4.0.0_3.0_1657809796803.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_ruperta_base_finetuned_spa_constitution","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_ruperta_base_finetuned_spa_constitution","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.roberta.base_finetuned").predict("""PUT YOUR STRING HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_ruperta_base_finetuned_spa_constitution|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|472.8 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/mrm8488/RuPERTa-base-finetuned-spa-constitution