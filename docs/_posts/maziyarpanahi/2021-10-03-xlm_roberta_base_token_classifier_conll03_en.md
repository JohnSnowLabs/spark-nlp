---
layout: model
title: XLM-RoBERTa Token Classification Base - NER CoNLL (xlm_roberta_base_token_classifier_conll03)
author: John Snow Labs
name: xlm_roberta_base_token_classifier_conll03
date: 2021-10-03
tags: [ner, conll, en, english, xlm_roberta, token_classification, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`XLM-RoBERTa Model` with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

**xlm_roberta_base_token_classifier_conll03** is a fine-tuned XLM-RoBERTa model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER), and Miscellaneous (MISC). 

We used [TFXLMRobertaForTokenClassification](https://huggingface.co/transformers/model_doc/xlmroberta.html#xlmrobertafortokenclassification) to train this model and used `XlmRoBertaForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_token_classifier_conll03_en_3.3.0_3.0_1633270922324.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
      .pretrained('xlm_roberta_base_token_classifier_conll03', 'en') \
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

## Results

```bash
+------------------------------------------------------------------------------------+
 |result                                                                              |
 +------------------------------------------------------------------------------------+
 |[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O, O, B-LOC]|
 +------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_token_classifier_conll03|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://www.clips.uantwerpen.be/conll2003/ner/](https://www.clips.uantwerpen.be/conll2003/ner/)

## Benchmarking

```bash
            precision    recall  f1-score   support

       B-LOC       0.95      0.84      0.89      1837
      B-MISC       0.87      0.86      0.86       922
       B-ORG       0.82      0.91      0.86      1341
       B-PER       0.96      0.96      0.96      1842
       I-LOC       0.89      0.84      0.86       257
      I-MISC       0.84      0.76      0.80       346
       I-ORG       0.88      0.91      0.90       751
       I-PER       0.98      0.97      0.97      1307
           O       0.99      1.00      1.00     42759

    accuracy                           0.98     51362
   macro avg       0.91      0.89      0.90     51362
weighted avg       0.98      0.98      0.98     51362



processed 51362 tokens with 5942 phrases; found: 5900 phrases; correct: 5257.
accuracy:  90.17%; (non-O)
accuracy:  98.08%; precision:  89.10%; recall:  88.47%; FB1:  88.79
              LOC: precision:  94.22%; recall:  83.34%; FB1:  88.45  1625
             MISC: precision:  84.40%; recall:  83.30%; FB1:  83.84  910
              ORG: precision:  79.60%; recall:  89.34%; FB1:  84.19  1505
              PER: precision:  94.62%; recall:  95.55%; FB1:  95.08  1860

```