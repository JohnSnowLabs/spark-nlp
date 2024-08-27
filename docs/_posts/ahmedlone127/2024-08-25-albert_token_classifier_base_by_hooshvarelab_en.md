---
layout: model
title: ALBERT Token Classification Base (albert_token_classifier_base_by_hooshvarelab)
author: John Snow Labs
name: albert_token_classifier_base_by_hooshvarelab
date: 2024-08-25
tags: [token_classification, albert, openvino, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: openvino
annotator: AlbertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

“
        
 ALBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_token_classifier_base_by_hooshvarelab_en_5.4.2_3.0_1724604140907.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_token_classifier_base_by_hooshvarelab_en_5.4.2_3.0_1724604140907.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() .setInputCol('text') .setOutputCol('document')

tokenizer = Tokenizer() .setInputCols(['document']) .setOutputCol('token')

tokenClassifier = AlbertForTokenClassification .pretrained('albert_token_classifier_base_by_hooshvarelab', 'en') .setInputCols(['token', 'document']) .setOutputCol('ner') .setCaseSensitive(True) .setMaxSentenceLength(512)

# since output column is IOB/IOB2 style, NerConverter can extract entities
ner_converter = NerConverter() .setInputCols(['document', 'token', 'ner']) .setOutputCol('entities')

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
tokenClassifier,
ner_converter
])

example = spark.createDataFrame([['My name is John!']]).toDF("text")
result = pipeline.fit(example).transform(example)

```
```scala

val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = AlbertForTokenClassification.pretrained("albert_token_classifier_base_by_hooshvarelab", "en")
.setInputCols("document", "token")
.setOutputCol("ner")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

// since output column is IOB/IOB2 style, NerConverter can extract entities
val ner_converter = NerConverter() 
.setInputCols("document", "token", "ner") 
.setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["My name is John!"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_token_classifier_base_by_hooshvarelab|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[label]|
|Language:|en|
|Size:|41.9 MB|
|Case sensitive:|true|