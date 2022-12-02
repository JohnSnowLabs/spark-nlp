---
layout: model
title: Ukrainian RoBERTa Embeddings
author: John Snow Labs
name: roberta_embeddings_ukr_roberta_base
date: 2022-04-14
tags: [roberta, embeddings, uk, open_source]
task: Embeddings
language: uk
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
recommended: true
annotator: RoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `ukr-roberta-base` is a Ukrainian model orginally trained by `youscan`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_ukr_roberta_base_uk_3.4.2_3.0_1649948817339.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_ukr_roberta_base","uk") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Я люблю Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_ukr_roberta_base","uk") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Я люблю Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("uk.embed.ukr_roberta_base").predict("""Я люблю Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_ukr_roberta_base|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|uk|
|Size:|474.2 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/youscan/ukr-roberta-base
- https://dumps.wikimedia.org/ukwiki/latest/ukwiki-latest-pages-articles.xml.bz2
- https://oscar-public.huma-num.fr/shuffled/uk_dedup.txt.gz
- https://github.com/youscan/language-models
- https://twitter.com/vitaliradchenko
