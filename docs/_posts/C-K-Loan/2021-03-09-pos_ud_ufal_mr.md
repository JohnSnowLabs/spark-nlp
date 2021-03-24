---
layout: model
title: Part of Speech for Marathi
author: John Snow Labs
name: pos_ud_ufal
date: 2021-03-09
tags: [part_of_speech, open_source, marathi, pos_ud_ufal, mr]
task: Part of Speech Tagging
language: mr
edition: Spark NLP 3.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A [Part of Speech](https://en.wikipedia.org/wiki/Part_of_speech) classifier predicts a grammatical label for every token in the input text. Implemented with an `averaged perceptron architecture`.

## Predicted Entities

- DET
- AUX
- NOUN
- PUNCT
- PRON
- ADJ
- CCONJ
- ADV
- VERB
- SCONJ
- NUM
- ADP
- INTJ
- PROPN

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_ufal_mr_3.0.0_3.0_1615292224912.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

sentence_detector = SentenceDetector()
  .setInputCols(["document"])
  .setOutputCol("sentence")

pos = PerceptronModel.pretrained("pos_ud_ufal", "mr")
  .setInputCols(["document", "token"])
  .setOutputCol("pos")

pipeline = Pipeline(stages=[
  document_assembler,
  sentence_detector,
  posTagger
])

example = spark.createDataFrame(pd.DataFrame({'text': ["जॉन हिम लॅब्समधून हॅलो! "]}))

result = pipeline.fit(example).transform(example)


```
```scala

val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

val sentence_detector = SentenceDetector()
        .setInputCols(["document"])
.setOutputCol("sentence")

val pos = PerceptronModel.pretrained("pos_ud_ufal", "mr")
        .setInputCols(Array("document", "token"))
        .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, pos))

val result = pipeline.fit(Seq.empty["जॉन हिम लॅब्समधून हॅलो! "].toDS.toDF("text")).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""जॉन हिम लॅब्समधून हॅलो! ""]
token_df = nlu.load('mr.pos').predict(text)
token_df
    
```
</div>

## Results

```bash
       token    pos
                   
0        जॉन  PROPN
1        हिम   NOUN
2  लॅब्समधून    ADJ
3       हॅलो   VERB
4          !  PUNCT
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_ufal|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[pos]|
|Language:|mr|