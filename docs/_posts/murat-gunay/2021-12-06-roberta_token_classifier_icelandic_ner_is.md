---
layout: model
title: Icelandic NER Model
author: John Snow Labs
name: roberta_token_classifier_icelandic_ner
date: 2021-12-06
tags: [icelandic, roberta, token_classifier, ner, is, open_source]
task: Named Entity Recognition
language: is
edition: Spark NLP 3.3.2
spark_version: 2.4
supported: true
annotator: RoBertaForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model, imported from Hugging Face, was fine-tuned on the MIM-GOLD-NER dataset for the Icelandic language, leveraging `Roberta` embeddings and using `RobertaForTokenClassification` for NER purposes.

## Predicted Entities

`Date`, `Location`, `Miscellaneous`, `Money`, `Organization`, `Percent`, `Person`, `Time`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_icelandic_ner_is_3.3.2_2.4_1638796728651.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_token_classifier_icelandic_ner", "is"))\
.setInputCols(["sentence",'token'])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
text = """Ég heiti Peter Fergusson. Ég hef búið í New York síðan í október 2011 og unnið hjá Tesla Motor og þénað 100K $ á ári."""
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

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_token_classifier_icelandic_ner", "is"))\
.setInputCols(Array("sentence","token"))\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(Array("sentence", "token", "ner"))\
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["Ég heiti Peter Fergusson. Ég hef búið í New York síðan í október 2011 og unnið hjá Tesla Motor og þénað 100K $ á ári."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("is.ner").predict("""Ég heiti Peter Fergusson. Ég hef búið í New York síðan í október 2011 og unnið hjá Tesla Motor og þénað 100K $ á ári.""")
```

</div>

## Results

```bash
+----------------+------------+
|chunk           |ner_label   |
+----------------+------------+
|Peter Fergusson |Person      |
|New York        |Location    |
|október 2011    |Date        |
|Tesla Motor     |Organization|
|100K $          |Money       |
+----------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_token_classifier_icelandic_ner|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|is|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

[https://huggingface.co/m3hrdadfi/icelandic-ner-roberta](https://huggingface.co/m3hrdadfi/icelandic-ner-roberta)

## Benchmarking

```bash
label      score
Macro-F1-Score   0.957209
Micro-F1-Score   0.951866
```
