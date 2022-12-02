---
layout: model
title: Toxic content classifier for German
author: John Snow Labs
name: distilbert_base_sequence_classifier_toxicity
date: 2022-02-16
tags: [toxic, distilbert, sequence_classifier, de, open_source]
task: Text Classification
language: de
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` ([link](https://huggingface.co/ml6team/distilbert-base-german-cased-toxic-comments)) and it's been trained on GermEval21 and IWG Hatespeech datasets for the German language, leveraging `Distil-BERT` embeddings and `DistilBertForSequenceClassification` for text classification purposes.

## Predicted Entities

`non_toxic`, `toxic`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_sequence_classifier_toxicity_de_3.4.0_3.0_1645021339319.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = DistilBertForSequenceClassification\
      .pretrained('distilbert_base_sequence_classifier_toxicity', 'de') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class')

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([["Natürlich kann ich von zuwanderern mehr erwarten. muss ich sogar. sie müssen die sprache lernen, sie müssen die gepflogenheiten lernen und sich in die gesellschaft einfügen. dass muss ich nicht weil ich mich schon in die gesellschaft eingefügt habe. egal wo du hin ziehst, nirgendwo wird dir soviel zucker in den arsch geblasen wie in deutschland."]]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = Tokenizer() 
    .setInputCols(Array("document")) 
    .setOutputCol("token")

val tokenClassifier = DistilBertForSequenceClassification.pretrained("distilbert_base_sequence_classifier_toxicity", "de")
      .setInputCols(Array("document", "token"))
      .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("Natürlich kann ich von zuwanderern mehr erwarten. muss ich sogar. sie müssen die sprache lernen, sie müssen die gepflogenheiten lernen und sich in die gesellschaft einfügen. dass muss ich nicht weil ich mich schon in die gesellschaft eingefügt habe. egal wo du hin ziehst, nirgendwo wird dir soviel zucker in den arsch geblasen wie in deutschland.").toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
['toxic']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_sequence_classifier_toxicity|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[class]|
|Language:|de|
|Size:|252.8 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- [GermEval21](https://github.com/germeval2021toxic/SharedTask/tree/main/Data%20Sets)
- [IWG Hatespeech](https://github.com/UCSM-DUE/IWG_hatespeech_public)
