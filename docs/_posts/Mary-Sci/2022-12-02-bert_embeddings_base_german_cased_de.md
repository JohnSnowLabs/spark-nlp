---
layout: model
title: German BertForMaskedLM Base Cased model
author: John Snow Labs
name: bert_embeddings_base_german_cased
date: 2022-12-02
tags: [de, open_source, bert_embeddings, bertformaskedlm]
task: Embeddings
language: de
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-german-cased` is a German model originally trained by HuggingFace.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_base_german_cased_de_4.2.4_3.0_1670017649516.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_base_german_cased","de") \
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
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_base_german_cased","de") 
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
|Model Name:|bert_embeddings_base_german_cased|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|de|
|Size:|409.6 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/bert-base-german-cased
- https://static.tildacdn.com/tild6438-3730-4164-b266-613634323466/german_bert.png
- https://github.com/deepset-ai/FARM/issues/60
- https://deepset.ai/german-bert
- https://thumb.tildacdn.com/tild3162-6462-4566-b663-376630376138/-/format/webp/Screenshot_from_2020.png
- https://thumb.tildacdn.com/tild6335-3531-4137-b533-313365663435/-/format/webp/deepset_checkpoints.png
- https://raw.githubusercontent.com/deepset-ai/FARM/master/docs/img/deepset_logo.png
- https://deepset.ai/german-bert
- https://github.com/deepset-ai/FARM
- https://github.com/deepset-ai/haystack/
- https://twitter.com/deepset_ai
- https://www.linkedin.com/company/deepset-ai/
- https://deepset.ai