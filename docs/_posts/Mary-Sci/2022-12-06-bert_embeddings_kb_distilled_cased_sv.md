---
layout: model
title: Swedish BertForMaskedLM Cased model (from Addedk)
author: John Snow Labs
name: bert_embeddings_kb_distilled_cased
date: 2022-12-06
tags: [sv, open_source, bert_embeddings, bertformaskedlm]
task: Embeddings
language: sv
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `kbbert-distilled-cased` is a Swedish model originally trained by `Addedk`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_kb_distilled_cased_sv_4.2.4_3.0_1670326799484.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_kb_distilled_cased","sv") \
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
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_kb_distilled_cased","sv") 
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
|Model Name:|bert_embeddings_kb_distilled_cased|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|sv|
|Size:|308.3 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/Addedk/kbbert-distilled-cased
- https://spraakbanken.gu.se/en/resources/gigaword
- https://github.com/AddedK/swedish-mbert-distillation/blob/main/azureML/pretrain_distillation.py
- https://kth.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=2&af=%5B%5D&searchType=UNDERGRADUATE&sortOrder2=title_sort_asc&language=en&pid=diva2%3A1698451&aq=%5B%5B%7B%22freeText%22%3A%22added+kina%22%7D%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=-6142
- https://arxiv.org/abs/2103.06418
- https://spraakbanken.gu.se/en/resources/gigaword