---
layout: model
title: Arabic BertForMaskedLM Base Cased model (from CAMeL-Lab)
author: John Snow Labs
name: bert_embeddings_base_arabic_camel_msa_eighth
date: 2022-12-02
tags: [ar, open_source, bert_embeddings, bertformaskedlm]
task: Embeddings
language: ar
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-arabic-camelbert-msa-eighth` is a Arabic model originally trained by `CAMeL-Lab`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_base_arabic_camel_msa_eighth_ar_4.2.4_3.0_1670016070157.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_base_arabic_camel_msa_eighth_ar_4.2.4_3.0_1670016070157.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_base_arabic_camel_msa_eighth","ar") \
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
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_base_arabic_camel_msa_eighth","ar") 
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
|Model Name:|bert_embeddings_base_arabic_camel_msa_eighth|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|ar|
|Size:|409.1 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa-eighth
- https://arxiv.org/abs/2103.06678
- https://github.com/CAMeL-Lab/CAMeLBERT
- https://catalog.ldc.upenn.edu/LDC2011T11
- http://www.abuelkhair.net/index.php/en/arabic/abu-el-khair-corpus
- https://vlo.clarin.eu/search;jsessionid=31066390B2C9E8C6304845BA79869AC1?1&q=osian
- https://archive.org/details/arwiki-20190201
- https://oscar-corpus.com/
- https://github.com/google-research/bert
- https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L286-L297
- https://github.com/CAMeL-Lab/camel_tools
- https://github.com/CAMeL-Lab/CAMeLBERT