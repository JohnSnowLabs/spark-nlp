---
layout: model
title: Chinese BertForMaskedLM Cased model (from uer)
author: John Snow Labs
name: bert_embeddings_chinese_roberta_l_6_h_256
date: 2022-12-02
tags: [zh, open_source, bert_embeddings, bertformaskedlm]
task: Embeddings
language: zh
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `chinese_roberta_L-6_H-256` is a Chinese model originally trained by `uer`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_chinese_roberta_l_6_h_256_zh_4.2.4_3.0_1670021717667.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_chinese_roberta_l_6_h_256","zh") \
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
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_chinese_roberta_l_6_h_256","zh") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(True)    
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, bert_loaded))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_chinese_roberta_l_6_h_256|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|zh|
|Size:|39.2 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/uer/chinese_roberta_L-6_H-256
- https://github.com/dbiir/UER-py/
- https://arxiv.org/abs/1909.05658
- https://arxiv.org/abs/1908.08962
- https://github.com/dbiir/UER-py/wiki/Modelzoo
- https://github.com/CLUEbenchmark/CLUECorpus2020/
- https://github.com/dbiir/UER-py/
- https://cloud.tencent.com/