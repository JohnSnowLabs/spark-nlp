---
layout: model
title: Bangla Bert Embeddings
author: John Snow Labs
name: bert_embeddings_bangla_bert_base
date: 2022-04-11
tags: [bert, embeddings, bn, open_source]
task: Embeddings
language: bn
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
recommended: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bangla-bert-base` is a Bangla model orginally trained by `sagorsarker`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_bangla_bert_base_bn_3.4.2_3.0_1649673290861.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_bangla_bert_base_bn_3.4.2_3.0_1649673290861.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_bangla_bert_base","bn") \
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

val embeddings = BertEmbeddings.pretrained("bert_embeddings_bangla_bert_base","bn") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("আমি স্পার্ক এনএলপি ভালোবাসি").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("bn.embed.bangala_bert").predict("""আমি স্পার্ক এনএলপি ভালোবাসি""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_bangla_bert_base|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|bn|
|Size:|617.6 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/sagorsarker/bangla-bert-base
- https://github.com/sagorbrur/bangla-bert
- https://arxiv.org/abs/1810.04805
- https://github.com/google-research/bert
- https://oscar-corpus.com/
- https://dumps.wikimedia.org/bnwiki/latest/
- https://github.com/sagorbrur/bnlp
- https://github.com/sagorbrur/bangla-bert
- https://github.com/google-research/bert
- https://twitter.com/mapmeld
- https://github.com/rezacsedu/Classification_Benchmarks_Benglai_NLP
- https://github.com/sagorbrur/bangla-bert/blob/master/notebook/bangla-bert-evaluation-classification-task.ipynb
- https://github.com/sagorbrur/bangla-bert/tree/master/evaluations/wikiann
- https://arxiv.org/abs/2012.14353
- https://arxiv.org/abs/2104.08613
- https://arxiv.org/abs/2107.03844
- https://arxiv.org/abs/2101.00204
- https://github.com/sagorbrur
- https://www.tensorflow.org/tfrc
- https://github.com/google-research/bert
