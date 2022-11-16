---
layout: model
title: Named Entity Recognition - CoNLL03 DeBERTa Base (nerdl_conll_deberta_base)
author: John Snow Labs
name: nerdl_conll_deberta_base
date: 2022-06-01
tags: [en, english, deberta, v3, base, conll, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: NerDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`nerdl_conll_deberta_base` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together. `nerdl_conll_deberta_base` model is trained with the `deberta_v3_base` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nerdl_conll_deberta_base_en_4.0.0_3.0_1654102358585.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = DeBertaEmbeddings.pretrained("deberta_v3_base", "en")\
      .setInputCols(["token", "document"])\
      .setOutputCol("embeddings")\
      .setCaseSensitive(True)\
      .setMaxSentenceLength(512)

ner_model = NerDLModel.pretrained('nerdl_conll_deberta_base', 'en') \
    .setInputCols(['document', 'token', 'embeddings']) \
    .setOutputCol('ner')

ner_converter = NerConverter() \
    .setInputCols(['document', 'token', 'ner']) \
    .setOutputCol('entities')

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    embeddings,
    ner_model,
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

val embeddings = DeBertaEmbeddings.pretrained("deberta_v3_base", "en")
    .setInputCols("document", "token") 
    .setOutputCol("embeddings")
    .setCaseSensitive(true)
    .setMaxSentenceLength(512)

val ner_model = NerDLModel.pretrained("nerdl_conll_deberta_base", "en") 
    .setInputCols("document"', "token", "embeddings") 
    .setOutputCol("ner")

val ner_converter = NerConverter() 
    .setInputCols("document", "token", "ner") 
    .setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, ner_model, ner_converter))

val example = Seq.empty["My name is John!"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```

{:.nlu-block}
```python
import nlu

text = ["My name is John!"]

ner_df = nlu.load('en.ner.nerdl_conll_deberta_base').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nerdl_conll_deberta_base|
|Type:|ner|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.2 MB|

## References

[https://www.clips.uantwerpen.be/conll2003/ner/](https://www.clips.uantwerpen.be/conll2003/ner/)

## Benchmarking

```bash
Test:

       label  precision    recall  f1-score   support
       B-LOC       0.92      0.93      0.93      1668
       I-ORG       0.87      0.94      0.90       835
      I-MISC       0.62      0.71      0.66       216
       I-LOC       0.82      0.91      0.87       257
       I-PER       0.99      0.99      0.99      1156
      B-MISC       0.83      0.81      0.82       702
       B-ORG       0.89      0.93      0.91      1661
       B-PER       0.97      0.97      0.97      1617
   micro-avg       0.91      0.93      0.92      8112
   macro-avg       0.86      0.90      0.88      8112
weighted-avg       0.91      0.93      0.92      8112

processed 46435 tokens with 5648 phrases; found: 5719 phrases; correct: 5194.
accuracy:  93.13%; (non-O)
accuracy:  98.18%; precision:  90.82%; recall:  91.96%; FB1:  91.39
              LOC: precision:  92.05%; recall:  92.99%; FB1:  92.51  1685
             MISC: precision:  80.73%; recall:  78.77%; FB1:  79.74  685
              ORG: precision:  87.97%; recall:  91.57%; FB1:  89.73  1729
              PER: precision:  96.85%; recall:  97.03%; FB1:  96.94  1620



Dev:                                                                                
       label  precision    recall  f1-score   support
       B-LOC       0.96      0.96      0.96      1837
       I-ORG       0.88      0.94      0.91       751
      I-MISC       0.90      0.76      0.83       346
       I-LOC       0.91      0.92      0.92       257
       I-PER       0.99      0.98      0.98      1307
      B-MISC       0.93      0.87      0.90       922
       B-ORG       0.90      0.95      0.92      1341
       B-PER       0.97      0.99      0.98      1842
   micro-avg       0.94      0.95      0.94      8603
   macro-avg       0.93      0.92      0.92      8603
weighted-avg       0.94      0.95      0.94      8603


processed 51362 tokens with 5942 phrases; found: 5985 phrases; correct: 5622.
accuracy:  94.65%; (non-O)
accuracy:  98.89%; precision:  93.93%; recall:  94.61%; FB1:  94.27
              LOC: precision:  95.88%; recall:  96.24%; FB1:  96.06  1844
             MISC: precision:  91.65%; recall:  85.68%; FB1:  88.57  862
              ORG: precision:  88.56%; recall:  93.51%; FB1:  90.97  1416
              PER: precision:  97.16%; recall:  98.26%; FB1:  97.71  1863
```