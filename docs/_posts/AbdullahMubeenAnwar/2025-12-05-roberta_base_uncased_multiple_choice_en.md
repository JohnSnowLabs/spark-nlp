---
layout: model
title: RoBerta Base Uncased Multiple Choice Fine-Tuned (ONNX)
author: John Snow Labs
name: roberta_base_uncased_multiple_choice
date: 2025-12-05
tags: [multiple_choice, en, open_source, onnx]
task: Question Answering
language: en
edition: Spark NLP 6.1.0
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaForMultipleChoice
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a roberta-large architecture fine-tuned on the RACE multiple-choice reading comprehension dataset. It achieves ~85% accuracy on RACE test sets (88% middle-school, 83.5% high-school)

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_uncased_multiple_choice_en_6.1.0_3.0_1764976628446.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_uncased_multiple_choice_en_6.1.0_3.0_1764976628446.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import MultiDocumentAssembler
from sparknlp.annotator import RoBertaForMultipleChoice
from pyspark.ml import Pipeline

document_assembler = MultiDocumentAssembler() \
    .setInputCols(["question", "choices"]) \
    .setOutputCols(["document_question", "document_choices"])

albert_for_multiple_choice = RoBertaForMultipleChoice()     .pretrained(f"roberta_base_uncased_multiple_choice", "en") \
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
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val documentAssembler = new MultiDocumentAssembler()
  .setInputCols(Array("question", "choices"))
  .setOutputCols(Array("document_question", "document_choices"))

val robertaForMC = RoBertaForMultipleChoice.pretrained("roberta_base_uncased_multiple_choice", "en")
  .setInputCols(Array("document_question", "document_choices"))
  .setOutputCol("answer")
  .setBatchSize(4)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  robertaForMC
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

+------------------------------------------------------------------------------------------+-------------------------------------+
|question                                                                                  |result                               |
+------------------------------------------------------------------------------------------+-------------------------------------+
|In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.|[It is eaten with a fork and a knife]|
|The Eiffel Tower is located in which country?                                             |[Germany]                            |
|Which animal is known as the king of the jungle?                                          |[ Tiger]                             |
|Water boils at what temperature?                                                          |[90°C]                               |
|Which planet is known as the Red Planet?                                                  |[ Venus]                             |
|Which language is primarily spoken in Brazil?                                             |[ English]                           |
|The Great Wall of China was built to protect against invasions from which group?          |[ The Romans]                        |
|Which chemical element has the symbol 'O'?                                                |[ Ozone]                             |
|Which continent is the Sahara Desert located in?                                          |[Asia]                               |
|Which artist painted the Mona Lisa?                                                       |[ Leonardo da Vinci]                 |
+------------------------------------------------------------------------------------------+-------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_uncased_multiple_choice|
|Compatibility:|Spark NLP 6.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|false|
|Max sentence length:|512|