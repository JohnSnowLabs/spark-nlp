---
layout: model
title: NER Model for German
author: John Snow Labs
name: xlm_roberta_large_token_classifier_conll03
date: 2021-12-25
tags: [german, xlm, roberta, ner, de, open_source]
task: Named Entity Recognition
language: de
edition: Spark NLP 3.3.4
spark_version: 2.4
supported: true
annotator: XlmRoBertaForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` and it's been fine-tuned for the German language, leveraging `XLM-RoBERTa` embeddings and `XlmRobertaForTokenClassification` for NER purposes.

## Predicted Entities

`PER`, `ORG`, `LOC`, `MISC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_conll03_de_3.3.4_2.4_1640443988652.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_large_token_classifier_conll03", "de"))\
.setInputCols(["sentence",'token'])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
text = """Ibser begann seine Karriere beim ASK Ebreichsdorf. 2004 wechselte er zu Admira Wacker Mödling, wo er auch in der Akademie spielte."""
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

val tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_large_token_classifier_conll03", "de"))\
.setInputCols(Array("sentence","token"))\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(Array("sentence", "token", "ner"))\
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["Ibser begann seine Karriere beim ASK Ebreichsdorf. 2004 wechselte er zu Admira Wacker Mödling, wo er auch in der Akademie spielte."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("de.ner.xlm").predict("""Ibser begann seine Karriere beim ASK Ebreichsdorf. 2004 wechselte er zu Admira Wacker Mödling, wo er auch in der Akademie spielte.""")
```

</div>

## Results

```bash
+----------------------+---------+
|chunk                 |ner_label|
+----------------------+---------+
|Ibser                 |PER      |
|ASK Ebreichsdorf      |ORG      |
|Admira Wacker Mödling |ORG      |
+----------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_large_token_classifier_conll03|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|de|
|Size:|1.8 GB|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

[https://huggingface.co/xlm-roberta-large-finetuned-conll03-german](https://huggingface.co/xlm-roberta-large-finetuned-conll03-german)
