---
layout: model
title: English BertEmbeddings Base Cased model (from BAAI)
author: John Snow Labs
name: bert_embeddings_bge_base
date: 2023-09-22
tags: [en, open_source, bert_embeddings, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.1.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bge-base-en` is a English model originally trained by `BAAI`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_bge_base_en_5.1.0_3.0_1695368493416.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_bge_base_en_5.1.0_3.0_1695368493416.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_bge_base","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, bert_loaded])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols(Array("text")) 
    .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_bge_base","en") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(true)    
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, bert_loaded))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_bge_base|
|Compatibility:|Spark NLP 5.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|259.0 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/BAAI/bge-base-en
- https://github.com/FlagOpen/FlagEmbedding
- https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md
- https://arxiv.org/pdf/2309.07597.pdf
- https://data.baai.ac.cn/details/BAAI-MTP
- https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md
- https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune#hard-negatives
- https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md
- https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list
- https://www.SBERT.net
- https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list
- https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md
- https://platform.openai.com/docs/guides/embeddings
- https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md
- https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
- https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/
- https://github.com/staoxiao/RetroMAE
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain
- https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md
- https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker
- https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker
- https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE
- https://paperswithcode.com/sota?task=Classification&dataset=MTEB+AmazonCounterfactualClassification+%28en%29