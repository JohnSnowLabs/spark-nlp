---
layout: model
title: Part of Speech Tagger Pretrained with Clinical Data
author: John Snow Labs
name: pos_clinical
date: 2021-03-29
tags: [pos, parts_of_speech, en, licensed]
task: Part of Speech Tagging
language: en
edition: Spark NLP 3.0.0
spark_version: 3.0
supported: true
annotator: PerceptronModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A [Part of Speech](https://en.wikipedia.org/wiki/Part_of_speech) classifier predicts a grammatical label for every token in the input text. Implemented with an `averaged perceptron architecture`. This model was trained on additional medical data.

## Predicted Entities

- PROPN
- PUNCT
- ADJ
- NOUN
- VERB
- DET
- ADP
- AUX
- PRON
- PART
- SCONJ
- NUM
- ADV
- CCONJ
- X
- INTJ
- SYM

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/pos_clinical_en_3.0.0_3.0_1617052315327.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler =  new DocumentAssembler().setInputCol("text").setOutputCol("document")
tokenizer          =  new Tokenizer().setInputCols("document").setOutputCol("token")
pos                =  PerceptronModel.pretrained("pos_clinical","en","clinical/models").setInputCols("token","document")
pipeline = Pipeline(stages=[document_assembler, tokenizer, pos])
df = spark.createDataFrame([['POS assigns each token in a sentence a grammatical label']], ["text"])
result = pipeline.fit(df).transform(df)
result.select("pos.result").show(false)
```
```scala
val document_assembler =  new DocumentAssembler().setInputCol("text").setOutputCol("document")
val tokenizer          =  new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
val pos                =  PerceptronModel.pretrained("pos_clinical","en","clinical/models").setInputCols("token","document")
val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, pos))
val df = Seq("POS assigns each token in a sentence a grammatical label").toDF("text")
val result = pipeline.fit(df).transform(df)
result.select("pos.result").show(false)
```

{:.nlu-block}
```python
nlu.load('pos.clinical').predict("POS assigns each token in a sentence a grammatical label")
```
</div>

## Results

```bash
+------------------------------------------+
|result                                    |
+------------------------------------------+
|[NN, NNS, PND, NN, II, DD, NN, DD, JJ, NN]|
+------------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_clinical|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[pos]|
|Language:|en|