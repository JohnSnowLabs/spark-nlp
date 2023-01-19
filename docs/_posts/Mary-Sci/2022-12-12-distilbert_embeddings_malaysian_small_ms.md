---
layout: model
title: Malay DistilBertForMaskedLM Small Cased model (from w11wo)
author: John Snow Labs
name: distilbert_embeddings_malaysian_small
date: 2022-12-12
tags: [ms, open_source, distilbert_embeddings, distilbertformaskedlm]
task: Embeddings
language: ms
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `malaysian-distilbert-small` is a Malay model originally trained by `w11wo`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_malaysian_small_ms_4.2.4_3.0_1670864988773.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_malaysian_small_ms_4.2.4_3.0_1670864988773.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

distilbert_loaded = DistilBertEmbeddings.pretrained("distilbert_embeddings_malaysian_small","ms") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(False)
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, distilbert_loaded])

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
 
val distilbert_loaded = DistilBertEmbeddings.pretrained("distilbert_embeddings_malaysian_small","ms") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(false)    
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, distilbert_loaded))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_embeddings_malaysian_small|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|ms|
|Size:|248.4 MB|
|Case sensitive:|false|

## References

- https://huggingface.co/w11wo/malaysian-distilbert-small
- https://arxiv.org/abs/1910.01108
- https://github.com/sgugger
- https://github.com/piegu/fastai-projects/blob/master/finetuning-English-GPT2-any-language-Portuguese-HuggingFace-fastaiv2.ipynb
- https://w11wo.github.io/