---
layout: model
title: Telugu XLMRoBerta Embeddings (from neuralspace-reverie)
author: John Snow Labs
name: xlmroberta_embeddings_indic_transformers_te_xlmroberta
date: 2022-05-13
tags: [te, open_source, xlm_roberta, embeddings]
task: Embeddings
language: te
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: XlmRoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XLMRoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `indic-transformers-te-xlmroberta` is a Telugu model orginally trained by `neuralspace-reverie`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_embeddings_indic_transformers_te_xlmroberta_te_3.4.4_3.0_1652439933257.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_embeddings_indic_transformers_te_xlmroberta_te_3.4.4_3.0_1652439933257.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_indic_transformers_te_xlmroberta","te") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["నేను స్పార్క్ NLP ని ప్రేమిస్తున్నాను"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_indic_transformers_te_xlmroberta","te") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("నేను స్పార్క్ NLP ని ప్రేమిస్తున్నాను").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_embeddings_indic_transformers_te_xlmroberta|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|te|
|Size:|505.2 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/neuralspace-reverie/indic-transformers-te-xlmroberta
- https://oscar-corpus.com/