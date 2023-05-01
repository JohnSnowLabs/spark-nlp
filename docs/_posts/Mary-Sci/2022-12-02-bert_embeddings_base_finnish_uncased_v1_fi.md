---
layout: model
title: Finnish BertForMaskedLM Base Uncased model (from TurkuNLP)
author: John Snow Labs
name: bert_embeddings_base_finnish_uncased_v1
date: 2022-12-02
tags: [fi, open_source, bert_embeddings, bertformaskedlm]
task: Embeddings
language: fi
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-finnish-uncased-v1` is a Finnish model originally trained by `TurkuNLP`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_base_finnish_uncased_v1_fi_4.2.4_3.0_1670017573662.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_base_finnish_uncased_v1_fi_4.2.4_3.0_1670017573662.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_base_finnish_uncased_v1","fi") \
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
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_base_finnish_uncased_v1","fi") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(True)    
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, bert_loaded))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("fi.embed.bert.uncased_base").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_base_finnish_uncased_v1|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|fi|
|Size:|467.4 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1
- http://dl.turkunlp.org/finbert/bert-base-finnish-cased-v1.zip
- http://dl.turkunlp.org/finbert/bert-base-finnish-uncased-v1.zip
- https://arxiv.org/abs/1912.07076
- https://github.com/google-research/bert
- https://github.com/google-research/bert/blob/master/multilingual.md
- https://raw.githubusercontent.com/TurkuNLP/FinBERT/master/img/yle-ylilauta-curves.png
- https://fasttext.cc/
- https://github.com/spyysalo/finbert-text-classification
- https://github.com/spyysalo/yle-corpus
- https://github.com/spyysalo/ylilauta-corpus
- https://arxiv.org/abs/1908.04212
- https://github.com/Traubert/FiNer-rules
- https://arxiv.org/pdf/1908.04212.pdf
- https://github.com/jouniluoma/keras-bert-ner
- https://github.com/mpsilfve/finer-data
- https://universaldependencies.org/
- https://github.com/spyysalo/bert-pos
- http://hdl.handle.net/11234/1-2837
- http://dl.turkunlp.org/finbert/bert-base-finnish-uncased.zip
- http://dl.turkunlp.org/finbert/bert-base-finnish-cased.zip