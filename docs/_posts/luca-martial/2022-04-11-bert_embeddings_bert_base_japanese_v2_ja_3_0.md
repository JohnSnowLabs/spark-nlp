---
layout: model
title: Japanese Bert Embeddings (Base, v2)
author: John Snow Labs
name: bert_embeddings_bert_base_japanese_v2
date: 2022-04-11
tags: [bert, embeddings, ja, open_source]
task: Embeddings
language: ja
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bert-base-japanese-v2` is a Japanese model orginally trained by `cl-tohoku`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_bert_base_japanese_v2_ja_3.4.2_3.0_1649674281970.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_bert_base_japanese_v2_ja_3.4.2_3.0_1649674281970.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_base_japanese_v2","ja") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["私はSpark NLPを愛しています"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_base_japanese_v2","ja") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("私はSpark NLPを愛しています").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ja.embed.bert_base_japanese_v2").predict("""私はSpark NLPを愛しています""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_bert_base_japanese_v2|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|ja|
|Size:|417.5 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/cl-tohoku/bert-base-japanese-v2
- https://github.com/google-research/bert
- https://pypi.org/project/unidic-lite/
- https://github.com/cl-tohoku/bert-japanese/tree/v2.0
- https://taku910.github.io/mecab/
- https://github.com/neologd/mecab-ipadic-neologd
- https://github.com/polm/fugashi
- https://github.com/polm/unidic-lite
- https://www.tensorflow.org/tfrc/
- https://creativecommons.org/licenses/by-sa/3.0/
- https://www.tensorflow.org/tfrc/