---
layout: model
title: DeBERTa Token Classification Base - NER CoNLL (deberta_v3_base_token_classifier_conll03)
author: John Snow Labs
name: deberta_v3_base_token_classifier_conll03
date: 2022-05-06
tags: [open_source, deberta, v3, token_classification, en, english, conll, ner]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: DeBertaForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

DeBertaForTokenClassification can load DeBERTA Models v2 and v3 with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

deberta_v3_base_token_classifier_conll03 is a fine-tuned DeBERTa model that is ready to be used for Token Classification task such as Named Entity Recognition and it achieves state-of-the-art performance.

We used TFDebertaV2ForTokenClassification to train this model and used DeBertaForTokenClassification annotator in Spark NLP 🚀 for prediction at scale!  This model has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER), and Miscellaneous (MISC).

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_v3_base_token_classifier_conll03_en_3.4.4_3.0_1651825757462.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = DeBertaForTokenClassification.pretrained("deberta_v3_base_token_classifier_conll03", "en")\ 
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

val tokenClassifier = DeBertaForTokenClassification.pretrained("deberta_v3_base_token_classifier_conll03", "en")
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


{:.nlu-block}
```python
import nlu
nlu.load("en.ner.debertav3_base.conll03").predict("""I really liked that movie!""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_v3_base_token_classifier_conll03|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|605.5 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

[https://huggingface.co/datasets/conll2003](https://huggingface.co/datasets/conll2003)