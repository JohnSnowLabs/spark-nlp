---
layout: model
title: Albert Multiple Choice Base Model (Uncased, ONNX)
author: John Snow Labs
name: albert_base_uncased_multiple_choice
date: 2025-11-20
tags: [albert, en, open_source, onnx]
task: Question Answering
language: en
edition: Spark NLP 6.1.0
spark_version: 3.0
supported: true
engine: onnx
annotator: AlbertForMultipleChoice
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A lightweight ALBERT-based model fine-tuned for multiple-choice question answering, exported in ONNX format

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_base_uncased_multiple_choice_en_6.1.0_3.0_1763650099206.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_base_uncased_multiple_choice_en_6.1.0_3.0_1763650099206.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import MultiDocumentAssembler
from sparknlp.annotator import AlbertForMultipleChoice
from pyspark.ml import Pipeline

document_assembler = MultiDocumentAssembler() \
    .setInputCols(["question", "choices"]) \
    .setOutputCols(["document_question", "document_choices"])

albert_for_multiple_choice = AlbertForMultipleChoice() \
    .pretrained(f"albert_base_uncased_multiple_choice", "en") \
    .setInputCols(["document_question", "document_choices"]) \
    .setOutputCol("answer") \
    .setBatchSize(4)

pipeline = Pipeline(stages=[
    document_assembler,
    albert_for_multiple_choice
])

data = spark.createDataFrame([
    ("In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.",
     "It is eaten with a fork and a knife, It is eaten while held in the hand."),
    ("The Eiffel Tower is located in which country?", "Germany, France, Italy"),
    ("Which animal is known as the king of the jungle?", "Lion, Elephant, Tiger, Leopard"),
    ("Water boils at what temperature?", "90°C, 120°C, 100°C"),
    ("Which planet is known as the Red Planet?", "Jupiter, Mars, Venus"),
    ("Which language is primarily spoken in Brazil?", "Spanish, Portuguese, English"),
    ("The Great Wall of China was built to protect against invasions from which group?",
     "The Greeks, The Romans, The Mongols, The Persians"),
    ("Which chemical element has the symbol 'O'?", "Oxygenm, Osmium, Ozone"),
    ("Which continent is the Sahara Desert located in?", "Asia, Africa, South America"),
    ("Which artist painted the Mona Lisa?", "Vincent van Gogh, Leonardo da Vinci, Pablo Picasso")
], ["question", "choices"])

model = pipeline.fit(data)
results = model.transform(data)

results.select("question", "choices", "answer.result").show(truncate=False)
```
```scala
import com.johnsnowlabs.nlp.base.MultiDocumentAssembler
import com.johnsnowlabs.nlp.annotators.classifier.dl.AlbertForMultipleChoice
import org.apache.spark.ml.Pipeline
import spark.implicits._

val documentAssembler = new MultiDocumentAssembler()
  .setInputCols(Array("question", "choices"))
  .setOutputCols(Array("document_question", "document_choices"))

val albertForMultipleChoice = AlbertForMultipleChoice.pretrained("albert_base_uncased_multiple_choice", "en")
  .setInputCols(Array("document_question", "document_choices"))
  .setOutputCol("answer")
  .setBatchSize(4)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  albertForMultipleChoice
))

val data = Seq(
  ("In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.",
    "It is eaten with a fork and a knife, It is eaten while held in the hand."),
  ("The Eiffel Tower is located in which country?", "Germany, France, Italy"),
  ("Which animal is known as the king of the jungle?", "Lion, Elephant, Tiger, Leopard"),
  ("Water boils at what temperature?", "90°C, 120°C, 100°C"),
  ("Which planet is known as the Red Planet?", "Jupiter, Mars, Venus"),
  ("Which language is primarily spoken in Brazil?", "Spanish, Portuguese, English"),
  ("The Great Wall of China was built to protect against invasions from which group?",
    "The Greeks, The Romans, The Mongols, The Persians"),
  ("Which chemical element has the symbol 'O'?", "Oxygenm, Osmium, Ozone"),
  ("Which continent is the Sahara Desert located in?", "Asia, Africa, South America"),
  ("Which artist painted the Mona Lisa?", "Vincent van Gogh, Leonardo da Vinci, Pablo Picasso")
).toDF("question", "choices")

val model = pipeline.fit(data)
val results = model.transform(data)

results.select("question", "choices", "answer.result").show(false)

```
</div>

## Results

```bash

+------------------------------------------------------------------------------------------+------------------------------------------------------------------------+--------------------------------------+
|question                                                                                  |choices                                                                 |result                                |
+------------------------------------------------------------------------------------------+------------------------------------------------------------------------+--------------------------------------+
|In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.|It is eaten with a fork and a knife, It is eaten while held in the hand.|[ It is eaten while held in the hand.]|
|The Eiffel Tower is located in which country?                                             |Germany, France, Italy                                                  |[ Italy]                              |
|Which animal is known as the king of the jungle?                                          |Lion, Elephant, Tiger, Leopard                                          |[ Elephant]                           |
|Water boils at what temperature?                                                          |90°C, 120°C, 100°C                                                      |[ 100°C]                              |
|Which planet is known as the Red Planet?                                                  |Jupiter, Mars, Venus                                                    |[ Mars]                               |
|Which language is primarily spoken in Brazil?                                             |Spanish, Portuguese, English                                            |[ Portuguese]                         |
|The Great Wall of China was built to protect against invasions from which group?          |The Greeks, The Romans, The Mongols, The Persians                       |[ The Mongols]                        |
|Which chemical element has the symbol 'O'?                                                |Oxygenm, Osmium, Ozone                                                  |[ Osmium]                             |
|Which continent is the Sahara Desert located in?                                          |Asia, Africa, South America                                             |[ South America]                      |
|Which artist painted the Mona Lisa?                                                       |Vincent van Gogh, Leonardo da Vinci, Pablo Picasso                      |[ Pablo Picasso]                      |
+------------------------------------------------------------------------------------------+------------------------------------------------------------------------+--------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_base_uncased_multiple_choice|
|Compatibility:|Spark NLP 6.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|44.2 MB|
|Case sensitive:|false|
|Max sentence length:|512|