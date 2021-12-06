---
layout: model
title: NER Model for 10 African Languages
author: John Snow Labs
name: xlm_roberta_large_token_classifier_masakhaner
date: 2021-12-06
tags: [amharic, hausa, igbo, kinyarwanda, luganda, swahilu, wolof, yoruba, token_classifier, xlm_roberta, ner, nigerian, pidgin, xx, open_source]
task: Named Entity Recognition
language: xx
edition: Spark NLP 3.3.2
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

- This model is imported from `Hugging Face`. 

- It's been trained using `xlm_roberta_large` fine-tuned model on 10 African languages (Amharic, Hausa, Igbo, Kinyarwanda, Luganda, Nigerian, Pidgin, Swahilu, Wolof, and Yorùbá).

## Predicted Entities

`DATE`, `LOC`, `PER`, `ORG`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_masakhaner_xx_3.3.2_2.4_1638784947143.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_large_token_classifier_masakhaner", "tr"))\
  .setInputCols(["sentence",'token'])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")
      
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
text = """አህመድ ቫንዳ ከ3-10-2000 ጀምሮ በአዲስ አበባ ኖሯል።"""
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

val tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_base_token_classifier_ner", "tr"))\
  .setInputCols(Array("sentence","token"))\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(Array("sentence", "token", "ner"))\
      .setOutputCol("ner_chunk")
      
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["አህመድ ቫንዳ ከ3-10-2000 ጀምሮ በአዲስ አበባ ኖሯል።"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
+----------------+---------+
|chunk           |ner_label|
+----------------+---------+
|አህመድ ቫንዳ      |PER      |
|ከ3-10-2000 ጀምሮ|DATE      |
|በአዲስ አበባ       |LOC      |
+----------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_large_token_classifier_masakhaner|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|xx|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

[https://huggingface.co/Davlan/xlm-roberta-large-masakhaner](https://huggingface.co/Davlan/xlm-roberta-large-masakhaner)

## Benchmarking

```bash
language:   F1-score:
--------    --------
amh	     75.76
hau	     91.75
ibo	     86.26
kin	     76.38
lug	     84.64
luo	     80.65
pcm	     89.55
swa	     89.48
wol	     70.70
yor	     82.05
```