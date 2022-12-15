---
layout: model
title: Spanish BertForMaskedLM Base Cased model (from dccuchile)
author: John Snow Labs
name: bert_embeddings_base_spanish_wwm_cased
date: 2022-12-02
tags: [es, open_source, bert_embeddings, bertformaskedlm]
task: Embeddings
language: es
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-spanish-wwm-cased` is a Spanish model originally trained by `dccuchile`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_base_spanish_wwm_cased_es_4.2.4_3.0_1670018860888.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_base_spanish_wwm_cased","es") \
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
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_base_spanish_wwm_cased","es") 
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
|Model Name:|bert_embeddings_base_spanish_wwm_cased|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|es|
|Size:|412.2 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased
- https://github.com/google-research/bert
- https://github.com/josecannete/spanish-corpora
- https://github.com/google-research/bert/blob/master/multilingual.md
- https://users.dcc.uchile.cl/~jperez/beto/uncased_2M/tensorflow_weights.tar.gz
- https://users.dcc.uchile.cl/~jperez/beto/uncased_2M/pytorch_weights.tar.gz
- https://users.dcc.uchile.cl/~jperez/beto/cased_2M/tensorflow_weights.tar.gz
- https://users.dcc.uchile.cl/~jperez/beto/cased_2M/pytorch_weights.tar.gz
- https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1827
- https://www.kaggle.com/nltkdata/conll-corpora
- https://github.com/gchaperon/beto-benchmarks/blob/master/conll2002/dev_results_beto-cased_conll2002.txt
- https://github.com/facebookresearch/MLDoc
- https://github.com/gchaperon/beto-benchmarks/blob/master/MLDoc/dev_results_beto-cased_mldoc.txt
- https://github.com/gchaperon/beto-benchmarks/blob/master/MLDoc/dev_results_beto-uncased_mldoc.txt
- https://github.com/google-research-datasets/paws/tree/master/pawsx
- https://github.com/facebookresearch/XNLI
- https://colab.research.google.com/drive/1uRwg4UmPgYIqGYY4gW_Nsw9782GFJbPt
- https://www.adere.so/
- https://imfd.cl/en/
- https://www.tensorflow.org/tfrc
- https://users.dcc.uchile.cl/~jperez/papers/pml4dc2020.pdf
- https://github.com/google-research/bert/blob/master/multilingual.md
- https://arxiv.org/pdf/1904.09077.pdf
- https://arxiv.org/pdf/1906.01502.pdf
- https://arxiv.org/abs/1812.10464
- https://arxiv.org/pdf/1901.07291.pdf
- https://arxiv.org/pdf/1904.02099.pdf
- https://arxiv.org/pdf/1906.01569.pdf
- https://arxiv.org/abs/1908.11828