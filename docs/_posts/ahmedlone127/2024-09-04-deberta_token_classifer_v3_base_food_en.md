---
layout: model
title: English DeBertaForTokenClassification (from davanstrien)
author: John Snow Labs
name: deberta_token_classifer_v3_base_food
date: 2024-09-04
tags: [token_classification, deberta, openvino, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: openvino
annotator: DeBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

“
DeBertaForTokenClassification can load DeBERTA Models v2 and v3 with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

deberta_token_classifer_v3_base_food is a fine-tuned DeBERTa model that is ready to be used for Token Classification task such as Named Entity Recognition and it achieves state-of-the-art performance.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_token_classifer_v3_base_food_en_5.5.0_3.0_1725494292416.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_token_classifer_v3_base_food_en_5.5.0_3.0_1725494292416.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler()\ 
.setInputCol("text")\
.setOutputCol("document")

tokenizer = Tokenizer()\
.setInputCols(['document'])\
.setOutputCol('token') 

tokenClassifier = DeBertaForTokenClassification.pretrained("deberta_token_classifer_v3_base_food", "en")\
.setInputCols(["document", "token"])\
.setOutputCol("ner")\
.setCaseSensitive(True)\ 
.setMaxSentenceLength(512) 

# since output column is IOB/IOB2 style, NerConverter can extract entities
ner_converter = NerConverter()\
.setInputCols(['document', 'token', 'ner'])\
.setOutputCol('entities') 

pipeline = Pipeline(stages=[
document_assembler,
tokenizer,
tokenClassifier,
ner_converter
])

example = spark.createDataFrame([['I really liked that movie!']]).toDF("text")
result = pipeline.fit(example).transform(example)



```
```scala

val document_assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = new Tokenizer()
.setInputCols("document")
.setOutputCol("token")

val tokenClassifier = DeBertaForTokenClassification.pretrained("deberta_token_classifer_v3_base_food", "en")
.setInputCols("document", "token")
.setOutputCol("ner")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

// since output column is IOB/IOB2 style, NerConverter can extract entities
val ner_converter = NerConverter() 
.setInputCols("document", "token", "ner") 
.setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier, ner_converter))

val example = Seq("I really liked that movie!").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_token_classifer_v3_base_food|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[label]|
|Language:|en|
|Size:|609.7 MB|
|Case sensitive:|true|