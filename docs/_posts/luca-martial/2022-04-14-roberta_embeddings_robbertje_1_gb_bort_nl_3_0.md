---
layout: model
title: Dutch RoBERTa Embeddings (Bort)
author: John Snow Labs
name: roberta_embeddings_robbertje_1_gb_bort
date: 2022-04-14
tags: [roberta, embeddings, nl, open_source]
task: Embeddings
language: nl
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: RoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `robbertje-1-gb-bort` is a Dutch model orginally trained by `DTAI-KULeuven`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_robbertje_1_gb_bort_nl_3.4.2_3.0_1649949030427.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_embeddings_robbertje_1_gb_bort_nl_3.4.2_3.0_1649949030427.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_robbertje_1_gb_bort","nl") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Ik hou van vonk nlp"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_robbertje_1_gb_bort","nl") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Ik hou van vonk nlp").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("nl.embed.robbertje_1_gb_bort").predict("""Ik hou van vonk nlp""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_robbertje_1_gb_bort|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|nl|
|Size:|172.8 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/DTAI-KULeuven/robbertje-1-gb-bort
- http://github.com/iPieter/robbert
- http://github.com/iPieter/robbertje
- https://www.clinjournal.org/clinj/article/view/131
- https://www.clin31.ugent.be
- https://arxiv.org/abs/2101.05716