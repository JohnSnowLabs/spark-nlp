---
layout: model
title: Detect Ticker Alias in Financial texts
author: John Snow Labs
name: roberta_token_classifier_ticker
date: 2021-12-27
tags: [ticker, finance, ner, roberta, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.3.4
spark_version: 2.4
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` and it's been trained on [Kaggle dataset](https://www.kaggle.com/omermetinn/tweets-about-the-top-companies-from-2015-to-2020), leveraging `RoBERTa` embeddings and `RobertaForTokenClassification` for NER purposes.

## Predicted Entities

`TICKER`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_TICKER/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_ticker_en_3.3.4_2.4_1640603190724.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_token_classifier_ticker", "en"))\
.setInputCols(["sentence",'token'])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
text = """I am going to buy 100 shares of MFST tomorrow."""
result = model.transform(spark.createDataFrame([[text]]).toDF("text"))
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_token_classifier_ticker", "en"))\
.setInputCols(Array("sentence","token"))\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(Array("sentence", "token", "ner"))\
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["I am going to buy 100 shares of MFST tomorrow."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.ner.stocks_ticker").predict("""I am going to buy 100 shares of MFST tomorrow.""")
```

</div>

## Results

```bash
+-----+---------+
|chunk|ner_label|
+-----+---------+
|MFST |TICKER   |
+-----+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_token_classifier_ticker|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|465.3 MB|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

[https://huggingface.co/Jean-Baptiste/roberta-ticker](https://huggingface.co/Jean-Baptiste/roberta-ticker)

## Benchmarking

```bash
Precision : 0.914157
Recall : 0.788824
F1-Score : 0.846878
```