---
layout: model
title: English XLMRobertaForTokenClassification Base Uncased model (from tner)
author: John Snow Labs
name: xlmroberta_ner_base_uncased_all_english
date: 2022-08-14
tags: [en, open_source, xlm_roberta, ner]
task: Named Entity Recognition
language: en
nav_key: models
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XLMRobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `xlm-roberta-base-uncased-all-english` is a English model originally trained by `tner`.

## Predicted Entities

`actor`, `time`, `corporation`, `ordinal number`, `cardinal number`, `restaurant`, `director`, `rna`, `geopolitical area`, `rating`, `protein`, `percent`, `product`, `plot`, `dna`, `disease`, `cell line`, `law`, `other`, `quote`, `date`, `soundtrack`, `origin`, `amenity`, `chemical`, `event`, `cuisine`, `dish`, `work of art`, `genre`, `cell type`, `location`, `language`, `quantity`, `award`, `character name`, `facility`, `relationship`, `organization`, `opinion`, `group`, `money`, `person`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_base_uncased_all_english_en_4.1.0_3.0_1660449703306.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_base_uncased_all_english_en_4.1.0_3.0_1660449703306.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

token_classifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_base_uncased_all_english","en") \
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
 
val token_classifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_base_uncased_all_english","en") 
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
|Model Name:|xlmroberta_ner_base_uncased_all_english|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|804.6 MB|
|Case sensitive:|false|
|Max sentence length:|256|

## References

- https://huggingface.co/tner/xlm-roberta-base-uncased-all-english
- https://github.com/asahi417/tner