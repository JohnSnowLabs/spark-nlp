---
layout: model
title: Indonesion NER Model
author: John Snow Labs
name: xlm_roberta_large_token_classification_ner
date: 2021-12-03
tags: [id, indonesian, xlm, roberta, xlm_roberta, ner, open_source]
task: Named Entity Recognition
language: id
edition: Spark NLP 3.3.3
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is imported from `Hugging Face-models`. It is the Indonesian fine-tuned of "xlm-roberta-large" language model.

## Predicted Entities

`CRD`, `DAT`, `ORD`, `ORG`, `PER`, `PRC`, `PRD`, `QTY`, `REG`, `TIM`, `WOA`, `EVT`, `FAC`, `GPE`, `LAN`, `LAW`, `LOC`, `MON`, `NOR`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classification_ner_id_3.3.3_2.4_1638535619269.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_large_token_classifier_ner", "id"))\
  .setInputCols(["sentence",'token'])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")
      
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
text = """Nama saya Sarah dan saya tinggal di London."""
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

val tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_large_token_classifier_ner", "id"))\
  .setInputCols(Array("sentence","token"))\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(Array("sentence", "token", "ner"))\
      .setOutputCol("ner_chunk")
      
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["Nama saya Sarah dan saya tinggal di London."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
+-------+---------+
|chunk  |ner_label|
+-------+---------+
|Sarah  |PER      |
|London |GPE      |
+-------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_large_token_classification_ner|
|Compatibility:|Spark NLP 3.3.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|id|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

[https://huggingface.co/cahya/xlm-roberta-large-indonesian-NER](https://huggingface.co/cahya/xlm-roberta-large-indonesian-NER)