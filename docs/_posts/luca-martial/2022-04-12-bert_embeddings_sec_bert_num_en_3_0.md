---
layout: model
title: Financial English BERT Embeddings (Number masking)
author: John Snow Labs
name: bert_embeddings_sec_bert_num
date: 2022-04-12
tags: [bert, embeddings, en, open_source, financial]
task: Embeddings
language: en
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Financial Pretrained BERT Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `sec-bert-num` is a English model orginally trained by `nlpaueb`. This model is the same as Bert Base but we replace every number token with a [NUM] pseudo-token handling all numeric expressions in a uniform manner, disallowing their fragmentation).

If you are interested in Financial Embeddings, take a look also at these two models:

[sec-base](https://nlp.johnsnowlabs.com/2022/04/12/bert_embeddings_sec_bert_base_en_3_0.html): Same as Bert Base but trained with financial documents.
[sec-shape](https://nlp.johnsnowlabs.com/2022/04/12/bert_embeddings_sec_bert_sh_en_3_0.html): Same as Bert sec-base but we replace numbers with pseudo-tokens that represent the number’s shape, so numeric expressions (of known shapes) are no longer fragmented, e.g., '53.2' becomes '[XX.X]' and '40,200.5' becomes '[XX,XXX.X]'.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_sec_bert_num_en_3.4.2_3.0_1649759295271.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_sec_bert_num_en_3.4.2_3.0_1649759295271.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_sec_bert_num","en") \
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

val embeddings = BertEmbeddings.pretrained("bert_embeddings_sec_bert_num","en") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.sec_bert_num").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_sec_bert_num|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|409.5 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/nlpaueb/sec-bert-num
- https://arxiv.org/abs/2203.06482
- http://nlp.cs.aueb.gr/
