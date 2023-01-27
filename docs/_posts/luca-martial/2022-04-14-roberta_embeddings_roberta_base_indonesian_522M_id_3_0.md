---
layout: model
title: Indonesian RoBERTa Embeddings (from cahya)
author: John Snow Labs
name: roberta_embeddings_roberta_base_indonesian_522M
date: 2022-04-14
tags: [roberta, embeddings, id, open_source]
task: Embeddings
language: id
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: RoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `roberta-base-indonesian-522M` is a Indonesian model orginally trained by `cahya`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_roberta_base_indonesian_522M_id_3.4.2_3.0_1649948345202.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_embeddings_roberta_base_indonesian_522M_id_3.4.2_3.0_1649948345202.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_roberta_base_indonesian_522M","id") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Saya suka percikan NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_roberta_base_indonesian_522M","id") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Saya suka percikan NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("id.embed.roberta_base_indonesian_522M").predict("""Saya suka percikan NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_roberta_base_indonesian_522M|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|id|
|Size:|473.6 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/cahya/roberta-base-indonesian-522M
- https://github.com/cahya-wirawan/indonesian-language-models/tree/master/Transformers