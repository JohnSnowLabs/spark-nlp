---
layout: model
title: Urdu RoBERTa Embeddings
author: John Snow Labs
name: roberta_embeddings_roberta_urdu_small
date: 2022-04-14
tags: [roberta, embeddings, ur, open_source]
task: Embeddings
language: ur
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

Pretrained RoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `roberta-urdu-small` is a Urdu model orginally trained by `urduhack`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_roberta_urdu_small_ur_3.4.2_3.0_1649948085721.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_roberta_urdu_small","ur") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["مجھے سپارک این ایل پی سے محبت ہے"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_roberta_urdu_small","ur") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("مجھے سپارک این ایل پی سے محبت ہے").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ur.embed.roberta_urdu_small").predict("""مجھے سپارک این ایل پی سے محبت ہے""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_roberta_urdu_small|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|ur|
|Size:|473.9 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/urduhack/roberta-urdu-small
- https://github.com/urduhack/urduhack/blob/master/LICENSE
- https://github.com/urduhack/urduhack
