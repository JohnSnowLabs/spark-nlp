---
layout: model
title: BERT Base Uncased Multiple Choice Fine-Tuned (ONNX)
author: John Snow Labs
name: bert_base_uncased_multiple_choice
date: 2025-11-04
tags: [multiple_choice, en, open_source, onnx]
task: Question Answering
language: en
edition: Spark NLP 6.1.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForMultipleChoice
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a fine-tuned version of bert-base-uncased on the None dataset. 

It achieves the following results on the evaluation set:

- Loss: 1.4499
- Accuracy: 0.535

Originally from: [irfanamal/bert_multiple_choice](https://huggingface.co/irfanamal/bert_multiple_choice)

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_uncased_multiple_choice_en_6.1.0_3.0_1762245877790.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_uncased_multiple_choice_en_6.1.0_3.0_1762245877790.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import MultiDocumentAssembler
from sparknlp.annotator import BertForMultipleChoice
from pyspark.ml import Pipeline

document_assembler = MultiDocumentAssembler() \
    .setInputCols(["question", "choices"]) \
    .setOutputCols(["document_question", "document_choices"])

bert_for_multiple_choice = BertForMultipleChoice() \
    .pretrained("bert_base_uncased_multiple_choice", "en") \
    .setInputCols(["document_question", "document_choices"]) \
    .setOutputCol("answer")

pipeline = Pipeline(stages=[
    document_assembler,
    bert_for_multiple_choice
])

testing_data = [
    ("In Italy, pizza served in formal settings is presented unsliced.",
     "It is eaten with a fork and a knife, It is eaten while held in the hand."),
    ("The Eiffel Tower is located in which country?", "Germany, France, Italy"),
    ("Which animal is known as the king of the jungle?", "Lion, Elephant, Tiger, Leopard"),
    ("Water boils at what temperature?", "90°C, 120°C, 100°C"),
    ("Which planet is known as the Red Planet?", "Jupiter, Mars, Venus"),
    ("Which language is primarily spoken in Brazil?", "Spanish, Portuguese, English"),
    ("The Great Wall of China was built to protect against which group?", "The Greeks, The Romans, The Mongols, The Persians"),
    ("Which chemical element has the symbol 'O'?", "Oxygen, Osmium, Ozone"),
    ("Which continent is the Sahara Desert located in?", "Asia, Africa, South America"),
    ("Which artist painted the Mona Lisa?", "Vincent van Gogh, Leonardo da Vinci, Pablo Picasso")
]

testing_df = spark.createDataFrame(testing_data, ["question", "choices"])

pipeline_model = pipeline.fit(testing_df)
pipeline_df = pipeline_model.transform(testing_df)

pipeline_df.select("question", "answer.result").show(truncate=False)
```
```scala
import com.johnsnowlabs.nlp.base.MultiDocumentAssembler
import com.johnsnowlabs.nlp.annotator.BertForMultipleChoice
import org.apache.spark.ml.Pipeline
import spark.implicits._

val documentAssembler = new MultiDocumentAssembler()
  .setInputCols("question", "choices")
  .setOutputCols("document_question", "document_choices")

val bertForMultipleChoice = BertForMultipleChoice
  .pretrained("bert_base_uncased_multiple_choice", "en")
  .setInputCols("document_question", "document_choices")
  .setOutputCol("answer")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  bertForMultipleChoice
))

val testingData = Seq(
  ("In Italy, pizza served in formal settings is presented unsliced.",
   "It is eaten with a fork and a knife, It is eaten while held in the hand."),
  ("The Eiffel Tower is located in which country?", "Germany, France, Italy"),
  ("Which animal is known as the king of the jungle?", "Lion, Elephant, Tiger, Leopard"),
  ("Water boils at what temperature?", "90°C, 120°C, 100°C"),
  ("Which planet is known as the Red Planet?", "Jupiter, Mars, Venus"),
  ("Which language is primarily spoken in Brazil?", "Spanish, Portuguese, English"),
  ("The Great Wall of China was built to protect against which group?", "The Greeks, The Romans, The Mongols, The Persians"),
  ("Which chemical element has the symbol 'O'?", "Oxygen, Osmium, Ozone"),
  ("Which continent is the Sahara Desert located in?", "Asia, Africa, South America"),
  ("Which artist painted the Mona Lisa?", "Vincent van Gogh, Leonardo da Vinci, Pablo Picasso")
)

val testingDF = testingData.toDF("question", "choices")

val pipelineModel = pipeline.fit(testingDF)
val pipelineDF = pipelineModel.transform(testingDF)

pipelineDF.select("question", "answer.result").show(false)
```
</div>

## Results

```bash

+-----------------------------------------------------------------+-------------------------------------+
|question                                                         |result                               |
+-----------------------------------------------------------------+-------------------------------------+
|In Italy, pizza served in formal settings is presented unsliced. |[It is eaten with a fork and a knife]|
|The Eiffel Tower is located in which country?                    |[Germany]                            |
|Which animal is known as the king of the jungle?                 |[Lion]                               |
|Water boils at what temperature?                                 |[90°C]                               |
|Which planet is known as the Red Planet?                         |[ Mars]                              |
|Which language is primarily spoken in Brazil?                    |[ Portuguese]                        |
|The Great Wall of China was built to protect against which group?|[ The Mongols]                       |
|Which chemical element has the symbol 'O'?                       |[Oxygen]                             |
|Which continent is the Sahara Desert located in?                 |[ Africa]                            |
|Which artist painted the Mona Lisa?                              |[Vincent van Gogh]                   |
+-----------------------------------------------------------------+-------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_uncased_multiple_choice|
|Compatibility:|Spark NLP 6.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|409.4 MB|
|Case sensitive:|false|
|Max sentence length:|512|