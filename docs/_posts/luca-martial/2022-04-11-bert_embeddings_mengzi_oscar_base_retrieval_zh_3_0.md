---
layout: model
title: Chinese Bert Embeddings (Base, COCO-ir dataset)
author: John Snow Labs
name: bert_embeddings_mengzi_oscar_base_retrieval
date: 2022-04-11
tags: [bert, embeddings, zh, open_source]
task: Embeddings
language: zh
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `mengzi-oscar-base-retrieval` is a Chinese model orginally trained by `Langboat`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_mengzi_oscar_base_retrieval_zh_3.4.2_3.0_1649670875498.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_mengzi_oscar_base_retrieval_zh_3.4.2_3.0_1649670875498.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_mengzi_oscar_base_retrieval","zh") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_mengzi_oscar_base_retrieval","zh") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("zh.embed.mengzi_oscar_base_retrieval").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_mengzi_oscar_base_retrieval|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|zh|
|Size:|383.7 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/Langboat/mengzi-oscar-base-retrieval
- https://arxiv.org/abs/2110.06696
- https://github.com/Langboat/Mengzi/blob/main/Mengzi-Oscar.md
- https://github.com/microsoft/Oscar/blob/master/INSTALL.md
- https://github.com/Langboat/Mengzi/blob/main/Mengzi-Oscar.md