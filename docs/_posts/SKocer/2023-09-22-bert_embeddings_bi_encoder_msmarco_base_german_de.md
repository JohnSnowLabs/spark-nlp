---
layout: model
title: German BertEmbeddings Base Cased model (from PM-AI)
author: John Snow Labs
name: bert_embeddings_bi_encoder_msmarco_base_german
date: 2023-09-22
tags: [de, open_source, bert_embeddings, onnx]
task: Embeddings
language: de
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

Pretrained BertEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bi-encoder_msmarco_bert-base_german` is a German model originally trained by `PM-AI`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_bi_encoder_msmarco_base_german_de_5.1.0_3.0_1695368809437.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_bi_encoder_msmarco_base_german_de_5.1.0_3.0_1695368809437.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_bi_encoder_msmarco_base_german","de") \
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
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_bi_encoder_msmarco_base_german","de") 
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
|Model Name:|bert_embeddings_bi_encoder_msmarco_base_german|
|Compatibility:|Spark NLP 5.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[bert]|
|Language:|de|
|Size:|409.7 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/PM-AI/bi-encoder_msmarco_bert-base_german
- https://github.com/UKPLab/sentence-transformers
- https://microsoft.github.io/msmarco/#ranking
- https://arxiv.org/abs/2108.13897
- https://openreview.net/forum?id=wCu6T5xFjeJ
- https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/
- https://github.com/beir-cellar/beir
- https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_msmarco_v3_margin_MSE.py
- https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz
- https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_msmarco_v3_margin_MSE.py%5D
- https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/README.md
- https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_msmarco_v3_margin_MSE.py
- https://arxiv.org/abs/2104.12741
- https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html#bm25
- https://en.th-wildau.de/
- https://senseaition.com/
- https://www.linkedin.com/in/herrphilipps
- https://efre.brandenburg.de/efre/de/
- https://www.senseaition.com
- https://www.th-wildau.de