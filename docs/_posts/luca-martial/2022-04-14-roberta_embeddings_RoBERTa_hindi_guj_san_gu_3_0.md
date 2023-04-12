---
layout: model
title: Gujarati RoBERTa Embeddings (from surajp)
author: John Snow Labs
name: roberta_embeddings_RoBERTa_hindi_guj_san
date: 2022-04-14
tags: [roberta, embeddings, gu, open_source]
task: Embeddings
language: gu
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: RoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `RoBERTa-hindi-guj-san` is a Gujarati model orginally trained by `surajp`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_RoBERTa_hindi_guj_san_gu_3.4.2_3.0_1649948207679.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_embeddings_RoBERTa_hindi_guj_san_gu_3.4.2_3.0_1649948207679.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_RoBERTa_hindi_guj_san","gu") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["મને સ્પાર્ક એનએલપી ગમે છે"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_RoBERTa_hindi_guj_san","gu") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("મને સ્પાર્ક એનએલપી ગમે છે").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("gu.embed.RoBERTa_hindi_guj_san").predict("""મને સ્પાર્ક એનએલપી ગમે છે""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_RoBERTa_hindi_guj_san|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|gu|
|Size:|252.1 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/surajp/RoBERTa-hindi-guj-san
- https://github.com/goru001/inltk
- https://www.kaggle.com/disisbig/hindi-wikipedia-articles-172k
- https://www.kaggle.com/disisbig/gujarati-wikipedia-articles
- https://www.kaggle.com/disisbig/sanskrit-wikipedia-articles
- https://twitter.com/parmarsuraj99
- https://www.linkedin.com/in/parmarsuraj99/