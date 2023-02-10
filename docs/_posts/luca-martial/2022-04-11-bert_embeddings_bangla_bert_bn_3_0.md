---
layout: model
title: Bangla Bert Embeddings (from Kowsher)
author: John Snow Labs
name: bert_embeddings_bangla_bert
date: 2022-04-11
tags: [bert, embeddings, bn, open_source]
task: Embeddings
language: bn
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bangla-bert` is a Bangla model orginally trained by `Kowsher`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_bangla_bert_bn_3.4.2_3.0_1649673360956.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_bangla_bert_bn_3.4.2_3.0_1649673360956.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_bangla_bert","bn") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["আমি স্পার্ক এনএলপি ভালোবাসি"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_bangla_bert","bn") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("আমি স্পার্ক এনএলপি ভালোবাসি").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("bn.embed.bangla_bert").predict("""আমি স্পার্ক এনএলপি ভালোবাসি""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_bangla_bert|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|bn|
|Size:|615.0 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/Kowsher/bangla-bert
- https://github.com/Kowsher/bert-base-bangla
- https://arxiv.org/abs/1810.04805
- https://github.com/google-research/bert
- https://www.kaggle.com/gakowsher/bangla-language-model-dataset
- https://ssrn.com/abstract=
- http://kowsher.org/