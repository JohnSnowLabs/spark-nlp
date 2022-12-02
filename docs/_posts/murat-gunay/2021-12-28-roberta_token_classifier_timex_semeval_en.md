---
layout: model
title: Detect Time-related Terminology
author: John Snow Labs
name: roberta_token_classifier_timex_semeval
date: 2021-12-28
tags: [timex, ner, roberta, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.3.4
spark_version: 2.4
supported: true
annotator: RoBertaForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` and it's been trained to detect time-related terminology, leveraging `RoBERTa` embeddings and `RobertaForTokenClassification` for NER purposes.

## Predicted Entities

`Period`, `Year`, `Calendar-Interval`, `Month-Of-Year`, `Day-Of-Month`, `Day-Of-Week`, `Hour-Of-Day`, `Minute-Of-Hour`, `Number`, `Second-Of-Minute`, `Time-Zone`, `Part-Of-Day`, `Season-Of-Year`, `AMPM-Of-Day`, `Part-Of-Week`, `Week-Of-Year`, `Two-Digit-Year`, `Sum`, `Difference`, `Union`, `Intersection`, `Every-Nth`, `This`, `Last`, `Next`, `Before`, `After`, `Between`, `NthFromStart`, `NthFromEnd`, `Frequency`, `Modifier`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_TIMEX_SEMEVAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_timex_semeval_en_3.3.4_2.4_1640679857852.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_token_classifier_timex_semeval", "en"))\
.setInputCols(["sentence",'token'])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
text = """Model training was started at 22:12C and it took 3 days from Tuesday to Friday."""
result = model.transform(spark.createDataFrame([[text]]).toDF("text"))
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_token_classifier_timex_semeval", "en"))\
.setInputCols(Array("sentence","token"))\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(Array("sentence", "token", "ner"))\
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["Model training was started at 22:12C and it took 3 days from Tuesday to Friday."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.ner.time").predict("""Model training was started at 22:12C and it took 3 days from Tuesday to Friday.""")
```

</div>

## Results

```bash
+-------+-----------------+
|chunk  |ner_label        |
+-------+-----------------+
|22:12C |Period           |
|3      |Number           |
|days   |Calendar-Interval|
|Tuesday|Day-Of-Week      |
|to     |Between          |
|Friday |Day-Of-Week      |
+-------+-----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_token_classifier_timex_semeval|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|439.5 MB|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

[https://huggingface.co/clulab/roberta-timex-semeval](https://huggingface.co/clulab/roberta-timex-semeval)