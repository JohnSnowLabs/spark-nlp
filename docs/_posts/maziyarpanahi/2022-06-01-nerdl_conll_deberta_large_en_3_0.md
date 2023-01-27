---
layout: model
title: Named Entity Recognition - CoNLL03 DeBERTa Large (nerdl_conll_deberta_large)
author: John Snow Labs
name: nerdl_conll_deberta_large
date: 2022-06-01
tags: [en, english, conll, deberta, v3, large, open_source]
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

`nerdl_conll_deberta_large` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together. `nerdl_conll_deberta_large` model is trained with the `deberta_v3_large` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nerdl_conll_deberta_large_en_4.0.0_3.0_1654103260615.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nerdl_conll_deberta_large_en_4.0.0_3.0_1654103260615.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = DeBertaEmbeddings.pretrained("deberta_v3_large", "en")\
      .setInputCols(["token", "document"])\
      .setOutputCol("embeddings")\
      .setCaseSensitive(True)\
      .setMaxSentenceLength(512)

ner_model = NerDLModel.pretrained('nerdl_conll_deberta_large', 'en') \
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

val embeddings = DeBertaEmbeddings.pretrained("deberta_v3_large", "en")
    .setInputCols("document", "token") 
    .setOutputCol("embeddings")
    .setCaseSensitive(true)
    .setMaxSentenceLength(512)

val ner_model = NerDLModel.pretrained("nerdl_conll_deberta_large", "en") 
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

ner_df = nlu.load('en.ner.nerdl_conll_deberta_large').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nerdl_conll_deberta_large|
|Type:|ner|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.9 MB|

## References

[https://www.clips.uantwerpen.be/conll2003/ner/](https://www.clips.uantwerpen.be/conll2003/ner/)

## Benchmarking

```bash
Test:

       label  precision    recall  f1-score   support
       B-LOC       0.94      0.93      0.93      1668
       I-ORG       0.88      0.94      0.91       835
      I-MISC       0.72      0.74      0.73       216
       I-LOC       0.86      0.89      0.88       257
       I-PER       0.99      0.99      0.99      1156
      B-MISC       0.84      0.83      0.83       702
       B-ORG       0.90      0.93      0.91      1661
       B-PER       0.98      0.97      0.97      1617
   micro-avg       0.92      0.93      0.93      8112
weighted-avg       0.92      0.93      0.93      8112


Dev:
                                                                                
       label  precision    recall  f1-score   support
       B-LOC       0.96      0.97      0.97      1837
       I-ORG       0.93      0.95      0.94       751
      I-MISC       0.91      0.82      0.86       346
       I-LOC       0.95      0.93      0.94       257
       I-PER       0.99      0.98      0.98      1307
      B-MISC       0.94      0.89      0.92       922
       B-ORG       0.93      0.95      0.94      1341
       B-PER       0.98      0.99      0.98      1842
   micro-avg       0.96      0.96      0.96      8603
weighted-avg       0.96      0.96      0.96      8603
```