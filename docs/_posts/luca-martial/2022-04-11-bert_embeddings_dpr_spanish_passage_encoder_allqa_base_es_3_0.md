---
layout: model
title: Spanish Bert Embeddings (Base, Pasage, Allqa)
author: John Snow Labs
name: bert_embeddings_dpr_spanish_passage_encoder_allqa_base
date: 2022-04-11
tags: [bert, embeddings, es, open_source]
task: Embeddings
language: es
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `dpr-spanish-passage_encoder-allqa-base` is a Spanish model orginally trained by `IIC`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_dpr_spanish_passage_encoder_allqa_base_es_3.4.2_3.0_1649671207246.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_dpr_spanish_passage_encoder_allqa_base_es_3.4.2_3.0_1649671207246.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_dpr_spanish_passage_encoder_allqa_base","es") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Me encanta chispa nlp"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_dpr_spanish_passage_encoder_allqa_base","es") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Me encanta chispa nlp").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.embed.dpr_spanish_passage_encoder_allqa_base").predict("""Me encanta chispa nlp""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_dpr_spanish_passage_encoder_allqa_base|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|es|
|Size:|412.4 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/IIC/dpr-spanish-passage_encoder-allqa-base
- https://arxiv.org/abs/2004.04906
- https://github.com/facebookresearch/DPR
- https://arxiv.org/abs/2004.04906
- https://paperswithcode.com/sota?task=text+similarity&dataset=squad_es