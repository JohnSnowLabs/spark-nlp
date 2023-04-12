---
layout: model
title: Persian Named Entity Recognition (from HooshvareLab)
author: John Snow Labs
name: roberta_ner_roberta_fa_zwnj_base_ner
date: 2022-05-03
tags: [roberta, ner, open_source, fa]
task: Named Entity Recognition
language: fa
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. `roberta-fa-zwnj-base-ner` is a Persian model orginally trained by `HooshvareLab`.

## Predicted Entities

`PRO`, `PCT`, `PER`, `ORG`, `DAT`, `TIM`, `EVE`, `FAC`, `LOC`, `MON`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_ner_roberta_fa_zwnj_base_ner_fa_3.4.2_3.0_1651594463153.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_ner_roberta_fa_zwnj_base_ner_fa_3.4.2_3.0_1651594463153.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer() \
.setInputCols("sentence") \
.setOutputCol("token")

tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_ner_roberta_fa_zwnj_base_ner","fa") \
.setInputCols(["sentence", "token"]) \
.setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["من عاشق جرقه nlp هستم"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer() 
.setInputCols(Array("sentence"))
.setOutputCol("token")

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_ner_roberta_fa_zwnj_base_ner","fa") 
.setInputCols(Array("sentence", "token")) 
.setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("من عاشق جرقه nlp هستم").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("fa.ner.roberta_fa_zwnj_base_ner").predict("""من عاشق جرقه nlp هستم""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_ner_roberta_fa_zwnj_base_ner|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|fa|
|Size:|442.7 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/HooshvareLab/roberta-fa-zwnj-base-ner
- https://github.com/HaniehP/PersianNER
- http://nsurl.org/2019-2/tasks/task-7-named-entity-recognition-ner-for-farsi/
- https://elisa-ie.github.io/wikiann/
- https://github.com/hooshvare/parsner/issues