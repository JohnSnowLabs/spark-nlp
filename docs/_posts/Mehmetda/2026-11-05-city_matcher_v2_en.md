---
layout: model
title: City Text Matcher
author: John Snow Labs
name: city_matcher_v2
date: 2026-11-05
tags: [en, licensed, clinical, city, textmatcher, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 6.3.0
spark_version: 3.4
supported: true
annotator: TextMatcherInternalModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts cities in clinical notes using rule-based TextMatcherInternal annotator.

Note: It is important to note that due to the nature of city names, it is quite common for them to be confused with personal names. Therefore, careful interpretation and verification of extracted entities may be required to ensure accuracy.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/city_matcher_v2_en_6.3.0_3.4_1793883830356.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/city_matcher_v2_en_6.3.0_3.4_1793883830356.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

text_matcher = TextMatcherInternalModel.pretrained("city_matcher","en","clinical/models") \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("city_name")\
    .setMergeOverlapping(True)

mathcer_pipeline = Pipeline().setStages([
    documentAssembler,
    sentenceDetector,
    tokenizer,
    text_matcher])

data = spark.createDataFrame([["""Name: Johnson, Alice, Record date: 2093-03-22, MR: 846275.
Dr. Emily Brown, IP 192.168.1.1.
She is a 55-year-old female who was admitted to the Global Hospital in Los Angeles for hip replacement on 03/22/93.
Patient's VIN: 2HGFA165X8H123456, SSN: 444-55-8888, Driver's license no: C789012D.
Phone: (212) 555-7890, 4321 Oak Street, New York City, USA, E-MAIL: alice.johnson@example.com.
Patient has traveled to Tokyo, Paris, and Sydney in the past year."""]]).toDF("text")

result = mathcer_pipeline.fit(data).transform(data)


```

{:.jsl-block}
```python
documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

text_matcher = medical.TextMatcherInternalModel.pretrained("city_matcher","en","clinical/models") \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("city_name")\
    .setMergeOverlapping(True)

mathcer_pipeline = nlp.Pipeline().setStages([
    documentAssembler,
    sentenceDetector,
    tokenizer,
    text_matcher])

data = spark.createDataFrame([["""Name: Johnson, Alice, Record date: 2093-03-22, MR: 846275.
Dr. Emily Brown, IP 192.168.1.1.
She is a 55-year-old female who was admitted to the Global Hospital in Los Angeles for hip replacement on 03/22/93.
Patient's VIN: 2HGFA165X8H123456, SSN: 444-55-8888, Driver's license no: C789012D.
Phone: (212) 555-7890, 4321 Oak Street, New York City, USA, E-MAIL: alice.johnson@example.com.
Patient has traveled to Tokyo, Paris, and Sydney in the past year."""]]).toDF("text")

result = mathcer_pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val text_matcher = TextMatcherInternalModel.pretrained("city_matcher","en","clinical/models")
    .setInputCols(Array("sentence","token"))
    .setOutputCol("city_name")
    .setMergeOverlapping(true)

val mathcer_pipeline = new Pipeline().setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        text_matcher))

val data = Seq("""Name: Johnson, Alice, Record date: 2093-03-22, MR: 846275.
Dr. Emily Brown, IP 192.168.1.1.
She is a 55-year-old female who was admitted to the Global Hospital in Los Angeles for hip replacement on 03/22/93.
Patient's VIN: 2HGFA165X8H123456, SSN: 444-55-8888, Driver's license no: C789012D.
Phone: (212) 555-7890, 4321 Oak Street, New York City, USA, E-MAIL: alice.johnson@example.com.
Patient has traveled to Tokyo, Paris, and Sydney in the past year.""").toDF("text")

val result = mathcer_pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------+-----+---+-----+
|        chunk|begin|end|label|
+-------------+-----+---+-----+
|  Los Angeles|  163|173| CITY|
|New York City|  331|343| CITY|
|        Tokyo|  410|414| CITY|
|        Paris|  417|421| CITY|
|       Sydney|  428|433| CITY|
+-------------+-----+---+-----+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|city_matcher_v2|
|Compatibility:|Spark NLP 6.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[city_name]|
|Language:|en|
|Size:|1.0 MB|
|Case sensitive:|false|