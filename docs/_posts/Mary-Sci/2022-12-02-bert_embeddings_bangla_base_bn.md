---
layout: model
title: Bangla BertForMaskedLM Base Cased model (from sagorsarker)
author: John Snow Labs
name: bert_embeddings_bangla_base
date: 2022-12-02
tags: [bn, open_source, bert_embeddings, bertformaskedlm]
task: Embeddings
language: bn
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bangla-bert-base` is a Bangla model originally trained by `sagorsarker`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_bangla_base_bn_4.2.4_3.0_1670015550585.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_bangla_base","bn") \
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
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_bangla_base","bn") 
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
|Model Name:|bert_embeddings_bangla_base|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|bn|
|Size:|617.5 MB|
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