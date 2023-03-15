---
layout: model
title: Latin RobertaForTokenClassification Large Cased model (from tner)
author: John Snow Labs
name: roberta_token_classifier_large_ontonotes5
date: 2023-03-01
tags: [la, open_source, roberta, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: la
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-large-ontonotes5` is a Latin model originally trained by `tner`.

## Predicted Entities

`NORP`, `FAC`, `QUANTITY`, `LOC`, `EVENT`, `CARDINAL`, `LANGUAGE`, `GPE`, `ORG`, `TIME`, `PERSON`, `WORK_OF_ART`, `DATE`, `PRODUCT`, `PERCENT`, `LAW`, `ORDINAL`, `MONEY`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_large_ontonotes5_la_4.3.0_3.0_1677703467254.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_large_ontonotes5_la_4.3.0_3.0_1677703467254.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = RobertaForTokenClassification.pretrained("roberta_token_classifier_large_ontonotes5","la") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

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
 
val tokenClassifier = RobertaForTokenClassification.pretrained("roberta_token_classifier_large_ontonotes5","la") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_token_classifier_large_ontonotes5|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|la|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/tner/roberta-large-ontonotes5
- https://github.com/asahi417/tner
- https://github.com/asahi417/tner
- https://aclanthology.org/2021.eacl-demos.7/
- https://paperswithcode.com/sota?task=Token+Classification&dataset=tner%2Fontonotes5