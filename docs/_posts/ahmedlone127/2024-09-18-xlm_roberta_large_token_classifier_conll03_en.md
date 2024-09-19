---
layout: model
title: XLM-RoBERTa Token Classification Large - NER CoNLL (xlm_roberta_base_token_classifier_conll03)
author: John Snow Labs
name: xlm_roberta_large_token_classifier_conll03
date: 2024-09-18
tags: [token_classification, xlm_roberta, openvino, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: openvino
annotator: XlmRoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

“

XLM-RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

xlm_roberta_base_token_classifier_conll03 is a fine-tuned XLM-RoBERTa model that is ready to use for Named Entity Recognition and achieves state-of-the-art performance for the NER task. This model has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER), and Miscellaneous (MISC).

We used TFXLMRobertaForTokenClassification to train this model and used XlmRoBertaForTokenClassification annotator in Spark NLP 🚀 for prediction at scale!

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_conll03_en_5.5.0_3.0_1726648473317.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_conll03_en_5.5.0_3.0_1726648473317.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

		
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

tokenClassifier = XlmRoBertaForTokenClassification \
      .pretrained('xlm_roberta_large_token_classifier_conll03', 'en') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('ner') \
      .setCaseSensitive(True) \
      .setMaxSentenceLength(512)

# since output column is IOB/IOB2 style, NerConverter can extract entities
ner_converter = NerConverter() \
    .setInputCols(['document', 'token', 'ner']) \
    .setOutputCol('entities')

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

val tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_base_token_classifier_conll03", "en")
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
|Model Name:|xlm_roberta_large_token_classifier_conll03|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[label]|
|Language:|en|
|Size:|1.8 GB|
|Case sensitive:|true|