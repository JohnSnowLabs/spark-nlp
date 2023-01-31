---
layout: model
title: NER Model for 6 Scandinavian Languages
author: John Snow Labs
name: bert_token_classifier_scandi_ner
date: 2021-12-09
tags: [danish, norwegian, swedish, icelandic, faroese, ner, xx, open_source]
task: Named Entity Recognition
language: xx
edition: Spark NLP 3.3.2
spark_version: 2.4
supported: true
recommended: true
annotator: BertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` and it's been fine-tuned for 6 Scandinavian languages (Danish, Norwegian-Bokmål, Norwegian-Nynorsk, Swedish, Icelandic, Faroese), leveraging `Bert` embeddings and `BertForTokenClassification` for NER purposes.

## Predicted Entities

`PER`, `ORG`, `LOC`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_SCANDINAVIAN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_scandi_ner_xx_3.3.2_2.4_1639044930234.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_scandi_ner_xx_3.3.2_2.4_1639044930234.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_scandi_ner", "xx"))\
.setInputCols(["sentence",'token'])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
text = """Hans er professor ved Statens Universitet, som ligger i København, og han er en rigtig københavner."""
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_scandi_ner", "xx"))\
.setInputCols(Array("sentence","token"))\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(Array("sentence", "token", "ner"))\
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["Hans er professor ved Statens Universitet, som ligger i København, og han er en rigtig københavner."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("xx.ner.scandinavian").predict("""Hans er professor ved Statens Universitet, som ligger i København, og han er en rigtig københavner.""")
```

</div>

## Results

```bash
+-------------------+---------+
|chunk              |ner_label|
+-------------------+---------+
|Hans               |PER      |
|Statens Universitet|ORG      |
|København          |LOC      |
|københavner        |MISC     |
+-------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_scandi_ner|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|xx|
|Size:|666.9 MB|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

[https://huggingface.co/saattrupdan/nbailab-base-ner-scandi](https://huggingface.co/saattrupdan/nbailab-base-ner-scandi)

## Benchmarking

```bash
languages :  F1 Score:
----------   --------
Danish       0.8744
Bokmål       0.9106
Nynorsk      0.9042
Swedish      0.8837
Icelandic    0.8861
Faroese      0.9022
```