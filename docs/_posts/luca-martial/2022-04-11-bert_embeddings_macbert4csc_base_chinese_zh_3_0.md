---
layout: model
title: Chinese Bert Embeddings (from shibing624)
author: John Snow Labs
name: bert_embeddings_macbert4csc_base_chinese
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

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `macbert4csc-base-chinese` is a Chinese model orginally trained by `shibing624`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_macbert4csc_base_chinese_zh_3.4.2_3.0_1649669240219.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_macbert4csc_base_chinese_zh_3.4.2_3.0_1649669240219.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_macbert4csc_base_chinese","zh") \
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

val embeddings = BertEmbeddings.pretrained("bert_embeddings_macbert4csc_base_chinese","zh") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("zh.embed.macbert4csc_base_chinese").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_macbert4csc_base_chinese|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|zh|
|Size:|384.0 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/shibing624/macbert4csc-base-chinese
- https://github.com/shibing624/pycorrector
- https://pan.baidu.com/s/1BV5tr9eONZCI0wERFvr0gQ
- http://nlp.ee.ncu.edu.tw/resource/csc.html
- https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml
- https://github.com/shibing624/pycorrector/tree/master/pycorrector/macbert
- https://arxiv.org/abs/2004.13922