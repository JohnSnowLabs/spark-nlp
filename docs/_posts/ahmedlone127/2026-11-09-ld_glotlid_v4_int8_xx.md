---
layout: model
title: GlotLID v4 Language Identification - 2,060 Labels (fastText, int8)
author: John Snow Labs
name: ld_glotlid_v4_int8
date: 2026-11-09
tags: [xx, open_source, onnx]
task: Language Detection
language: xx
edition: Spark NLP 7.0.1
spark_version: 3.4
supported: true
engine: onnx
annotator: LanguageDetectorDL
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained LanguageDetectorDL model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. ld_glotlid_v4_int8 is the int8 version of ld_glotlid_v4 (GlotLID v4, a fastText language identification model originally trained by cis-lmu, exported to ONNX). Its embedding table is quantized to 8 bits, so it is 111 MB instead of 736 MB.

On 452k test texts it gave the same top label as the full model 99.8% of the time, and none of the differences were on texts where the full model's score was 0.7 or higher. It returns the same 2,060 labels (an ISO 639-3 code plus a script, such as `eng_Latn` or `cmn_Hani`) with the same settings: `unk` below the 0.7 threshold, `scriptPooling` on, `minLetters` set to 20 and the top 5 scores in the metadata. It works best on single sentences, so split documents into sentences first.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_glotlid_v4_int8_xx_7.0.1_3.4_1794264061864.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ld_glotlid_v4_int8_xx_7.0.1_3.4_1794264061864.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

language_detector = LanguageDetectorDL.pretrained("ld_glotlid_v4_int8", "xx") \
    .setInputCols(["sentence"]) \
    .setOutputCol("language")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    language_detector
])

data = spark.createDataFrame([
    ["The patient was admitted with chest pain and shortness of breath."],
    ["患者因胸痛和呼吸困难三天入院。"],
    ["El paciente ingresó por dolor torácico y dificultad para respirar."]
]).toDF("text")

result = pipeline.fit(data).transform(data)

result.select("text", "language.result").show(truncate=False)
```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols(Array("document"))
  .setOutputCol("sentence")

val languageDetector = LanguageDetectorDL.pretrained("ld_glotlid_v4_int8", "xx")
  .setInputCols(Array("sentence"))
  .setOutputCol("language")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  languageDetector
))

val data = Seq(
  "The patient was admitted with chest pain and shortness of breath.",
  "患者因胸痛和呼吸困难三天入院。",
  "El paciente ingresó por dolor torácico y dificultad para respirar."
).toDF("text")

val result = pipeline.fit(data).transform(data)

result.select("text", "language.result").show(truncate = false)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ld_glotlid_v4_int8|
|Compatibility:|Spark NLP 7.0.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[language]|
|Language:|xx|
|Size:|111.1 MB|