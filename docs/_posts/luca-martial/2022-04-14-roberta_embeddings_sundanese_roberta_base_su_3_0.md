---
layout: model
title: Sundanese RoBERTa Embeddings (from w11wo)
author: John Snow Labs
name: roberta_embeddings_sundanese_roberta_base
date: 2022-04-14
tags: [roberta, embeddings, su, open_source]
task: Embeddings
language: su
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: RoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `sundanese-roberta-base` is a Sundanese model orginally trained by `w11wo`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_sundanese_roberta_base_su_3.4.2_3.0_1649948770581.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_embeddings_sundanese_roberta_base_su_3.4.2_3.0_1649948770581.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_sundanese_roberta_base","su") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Abdi bogoh Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_sundanese_roberta_base","su") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Abdi bogoh Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("su.embed.sundanese_roberta_base").predict("""Abdi bogoh Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_sundanese_roberta_base|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|su|
|Size:|468.5 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/w11wo/sundanese-roberta-base
- https://arxiv.org/abs/1907.11692
- https://hf.co/datasets/oscar
- https://hf.co/datasets/mc4
- https://hf.co/datasets/cc100
- https://su.wikipedia.org/
- https://hf.co/w11wo/sundanese-roberta-base/tree/main
- https://hf.co/w11wo/sundanese-roberta-base/tensorboard
- https://w11wo.github.io/