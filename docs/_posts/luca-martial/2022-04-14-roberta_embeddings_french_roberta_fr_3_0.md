---
layout: model
title: French RoBERTa Embeddings (from abhilash1910)
author: John Snow Labs
name: roberta_embeddings_french_roberta
date: 2022-04-14
tags: [roberta, embeddings, fr, open_source]
task: Embeddings
language: fr
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: RoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `french-roberta` is a French model orginally trained by `abhilash1910`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_french_roberta_fr_3.4.2_3.0_1649947893823.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_embeddings_french_roberta_fr_3.4.2_3.0_1649947893823.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_french_roberta","fr") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["J'adore Spark Nlp"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_french_roberta","fr") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("J'adore Spark Nlp").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("fr.embed.french_roberta").predict("""J'adore Spark Nlp""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_french_roberta|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|fr|
|Size:|255.9 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/abhilash1910/french-roberta