---
layout: model
title: MPNet Sequence Classification - UKR Message Classifier
author: John Snow Labs
name: mpnet_sequence_classifier_ukr_message
date: 2024-01-10
tags: [en, mpnet, sequence, classification, open_source, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.2.3
spark_version: 3.0
supported: true
engine: onnx
annotator: MPNetForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

MPNet Sequence Classification imported from huggingface. 

Originally a SetFit model, reference: https://huggingface.co/rodekruis/sml-ukr-message-classifier

## Predicted Entities

`ANOMALY`, `ARMY`, `CHILDREN`, `CONNECTIVITY`, `CONNECTWITHREDCROSS`, `EDUCATION`, `FOOD`, `GOODSSERVICES`, `HEALTH`, `INCLUSIONCVA`, `LEGAL`, `MONEY/BANKING`, `NFINONFOODITEMS`, `OTHERPROGRAMSOTHERNGOS`, `PARCEL`, `PAYMENTCVA`, `PETS`, `PMER/NEWPROGRAMOPERTUNITIES`, `PROGRAMINFO`, `PROGRAMINFORMATION`, `PSSRFL`, `REGISTRATIONCVA`, `SENTIMENT/FEEDBACK`, `SHELTER`, `TRANSLATION/LANGUAGE`, `TRANSPORT/CAR`, `TRANSPORT/MOVEMENT`, `WASH`, `WORK/JOBS`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mpnet_sequence_classifier_ukr_message_en_5.2.3_3.0_1704907644396.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mpnet_sequence_classifier_ukr_message_en_5.2.3_3.0_1704907644396.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
document = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")
sequenceClassifier = MPNetForSequenceClassification \
    .pretrained() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("label")
data = spark.createDataFrame([
    ["I love driving my car."],
    ["The next bus will arrive in 20 minutes."],
    ["pineapple on pizza is the worst 🤮"],
]).toDF("text")
pipeline = Pipeline().setStages([document, tokenizer, sequenceClassifier])
pipelineModel = pipeline.fit(data)
results = pipelineModel.transform(data)
results.select("label.result").show()
```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val document = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val modelPath = "onnx_exported/rodekruis/sml-ukr-message-classifier"

val sequenceClassifier = MPNetForSequenceClassification
  .loadSavedModel(modelPath, spark)
//        .pretrained()
  .setInputCols(Array("document", "token"))
  .setOutputCol("label")

val texts: Seq[String] = Seq(
  "I love driving my car.",
  "The next bus will arrive in 20 minutes.",
  "pineapple on pizza is the worst 🤮")
val data = texts.toDF("text")

val pipeline = new Pipeline().setStages(Array(document, tokenizer, sequenceClassifier))
val pipelineModel = pipeline.fit(data)
val results = pipelineModel.transform(data)

results.select("label.result").show()
```
</div>

## Results

```bash
+--------------------+
|              result|
+--------------------+
|     [TRANSPORT/CAR]|
|[TRANSPORT/MOVEMENT]|
|              [FOOD]|
+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mpnet_sequence_classifier_ukr_message|
|Compatibility:|Spark NLP 5.2.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[label]|
|Language:|en|
|Size:|403.5 MB|