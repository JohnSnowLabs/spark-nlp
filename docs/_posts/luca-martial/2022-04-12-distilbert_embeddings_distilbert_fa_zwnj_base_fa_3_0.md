---
layout: model
title: Persian DistilBERT Embeddings (from HooshvareLab)
author: John Snow Labs
name: distilbert_embeddings_distilbert_fa_zwnj_base
date: 2022-04-12
tags: [distilbert, embeddings, fa, open_source]
task: Embeddings
language: fa
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: DistilBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBERT Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `distilbert-fa-zwnj-base` is a Persian model orginally trained by `HooshvareLab`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_distilbert_fa_zwnj_base_fa_3.4.2_3.0_1649783880670.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_distilbert_fa_zwnj_base","fa") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["من عاشق جرقه NLP هستم"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_distilbert_fa_zwnj_base","fa") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("من عاشق جرقه NLP هستم").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("fa.embed.distilbert_fa_zwnj_base").predict("""من عاشق جرقه NLP هستم""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_embeddings_distilbert_fa_zwnj_base|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|fa|
|Size:|282.6 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/HooshvareLab/distilbert-fa-zwnj-base
- https://github.com/hooshvare/parsbert/issues