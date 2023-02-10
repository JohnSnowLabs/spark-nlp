---
layout: model
title: Named Entity Recognition - CoNLL03 ELMO Base (nerdl_conll_elmo)
author: John Snow Labs
name: nerdl_conll_elmo
date: 2022-06-01
tags: [ner, en, english, elmo, conll, open_source]
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

`nerdl_conll_elmo` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together. `nerdl_conll_elmo` model is trained with `elmo` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nerdl_conll_elmo_en_4.0.0_3.0_1654103884644.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nerdl_conll_elmo_en_4.0.0_3.0_1654103884644.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = ElmoEmbeddings\
      .pretrained('elmo', 'en')\
      .setInputCols(["token", "document"])\
      .setOutputCol("embeddings")\
      .setPoolingLayer("elmo")

ner_model = NerDLModel.pretrained('nerdl_conll_elmo', 'en') \
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

val embeddings = ElmoEmbeddings.pretrained("elmo", "en")
    .setInputCols("document", "token") 
    .setOutputCol("embeddings")
    .setPoolingLayer("elmo")

val ner_model = NerDLModel.pretrained("nerdl_conll_elmo", "en") 
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

ner_df = nlu.load('en.ner.nerdl_conll_elmo').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nerdl_conll_elmo|
|Type:|ner|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|17.0 MB|

## References

[https://www.clips.uantwerpen.be/conll2003/ner/](https://www.clips.uantwerpen.be/conll2003/ner/)

## Benchmarking

```bash
Test:   

       label  precision    recall  f1-score   support
       B-LOC      0.941     0.932     0.936      1668
       I-ORG      0.886     0.932     0.908       835
      I-MISC      0.709     0.745     0.727       216
       I-LOC      0.863     0.879     0.871       257
       I-PER      0.974     0.993     0.983      1156
      B-MISC      0.843     0.825     0.834       702
       B-ORG      0.907     0.931     0.919      1661
       B-PER      0.961     0.972     0.966      1617
   micro-avg      0.920     0.932     0.926      8112
   macro-avg      0.885     0.901     0.893      8112
weighted-avg      0.920     0.932     0.926      8112

processed 46435 tokens with 5648 phrases; found: 5685 phrases; correct: 5209.
accuracy:  93.24%; (non-O)
accuracy:  98.30%; precision:  91.63%; recall:  92.23%; FB1:  91.93
              LOC: precision:  93.58%; recall:  92.69%; FB1:  93.13  1652
             MISC: precision:  82.03%; recall:  80.63%; FB1:  81.32  690
              ORG: precision:  89.46%; recall:  91.93%; FB1:  90.68  1707
              PER: precision:  95.97%; recall:  97.09%; FB1:  96.53  1636


Dev:
                                                                                
       label  precision    recall  f1-score   support
       B-LOC      0.974     0.968     0.971      1837
       I-ORG      0.923     0.948     0.936       751
      I-MISC      0.933     0.841     0.884       346
       I-LOC      0.948     0.930     0.939       257
       I-PER      0.981     0.984     0.982      1307
      B-MISC      0.939     0.903     0.921       922
       B-ORG      0.923     0.950     0.936      1341
       B-PER      0.972     0.981     0.976      1842
   micro-avg      0.956     0.955     0.956      8603
   macro-avg      0.949     0.938     0.943      8603
weighted-avg      0.956     0.955     0.956      8603

processed 51362 tokens with 5942 phrases; found: 5961 phrases; correct: 5651.
accuracy:  95.55%; (non-O)
accuracy:  99.07%; precision:  94.80%; recall:  95.10%; FB1:  94.95
              LOC: precision:  96.99%; recall:  96.46%; FB1:  96.72  1827
             MISC: precision:  92.13%; recall:  88.94%; FB1:  90.51  890
              ORG: precision:  91.05%; recall:  94.03%; FB1:  92.52  1385
              PER: precision:  96.72%; recall:  97.61%; FB1:  97.16  1859
```