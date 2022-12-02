---
layout: model
title: Hindi XLMRobertaForTokenClassification Large Cased model (from cfilt)
author: John Snow Labs
name: xlmroberta_ner_hiner_original_large
date: 2022-08-13
tags: [hi, open_source, xlm_roberta, ner]
task: Named Entity Recognition
language: hi
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XLMRobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `HiNER-original-xlm-roberta-large` is a Hindi model originally trained by `cfilt`.

## Predicted Entities

`GAME`, `MISC`, `ORGANIZATION`, `FESTIVAL`, `LOCATION`, `LITERATURE`, `LANGUAGE`, `NUMEX`, `PERSON`, `RELIGION`, `TIMEX`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_hiner_original_large_hi_4.1.0_3.0_1660406213542.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

token_classifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_hiner_original_large","hi") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")
    
ner_converter = NerConverter()\
    .setInputCols(["document", "token", "ner"])\
    .setOutputCol("ner_chunk") 
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, token_classifier, ner_converter])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val token_classifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_hiner_original_large","hi") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("document", "token', "ner"))
    .setOutputCol("ner_chunk")
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, token_classifier, ner_converter))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_ner_hiner_original_large|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|hi|
|Size:|1.8 GB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/cfilt/HiNER-original-xlm-roberta-large
- https://paperswithcode.com/sota?task=Token+Classification&dataset=HiNER+Original