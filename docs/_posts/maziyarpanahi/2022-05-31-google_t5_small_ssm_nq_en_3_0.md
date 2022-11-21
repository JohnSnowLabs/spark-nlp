---
layout: model
title: Google's T5 for closed book question answering
author: John Snow Labs
name: google_t5_small_ssm_nq
date: 2022-05-31
tags: [t5, en, question_answering, open_source]
task: Text Generation
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: T5Transformer
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a text-to-text model trained by Google on the colossal, cleaned version of Common Crawl's web crawl corpus (C4) data set and then fined tuned on Wikipedia and the natural questions (NQ) dataset. The model can answer free text questions, such as "Which is the capital of France ?" without relying on any context or external resources.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/google_t5_small_ssm_nq_en_4.0.0_3.0_1654004516744.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.annotator import SentenceDetectorDLModel, T5Transformer

data = self.spark.createDataFrame([
[1, "Which is the capital of France? Who was the first president of USA?"],
[1, "Which is the capital of Bulgaria ?"],
[2, "Who is Donald Trump?"]]).toDF("id", "text")

document_assembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("documents")

sentence_detector = SentenceDetectorDLModel\
.pretrained()\
.setInputCols(["documents"])\
.setOutputCol("questions")

t5 = T5Transformer()\
.pretrained("google_t5_small_ssm_nq")\
.setInputCols(["questions"])\
.setOutputCol("answers")\

pipeline = Pipeline().setStages([document_assembler, sentence_detector, t5])
results = pipeline.fit(data).transform(data)

results.select("questions.result", "answers.result").show(truncate=False)
```
```scala
val testData = ResourceHelper.spark.createDataFrame(Seq(

(1, "Which is the capital of France? Who was the first president of USA?"),
(1, "Which is the capital of Bulgaria ?"),
(2, "Who is Donald Trump?")

)).toDF("id", "text")

val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("documents")

val sentenceDetector = SentenceDetectorDLModel
.pretrained()
.setInputCols(Array("documents"))
.setOutputCol("questions")

val t5 = T5Transformer
.pretrained("google_t5_small_ssm_nq")
.setInputCols(Array("questions"))
.setOutputCol("answers")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, t5))

val model = pipeline.fit(testData)
val results = model.transform(testData)

results.select("questions.result", "answers.result").show(truncate = false)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.t5").predict("""Which is the capital of France? Who was the first president of USA?""")
```

</div>

## Results

```bash
+-------------------------------------------------------------------------------------------------------------+-----------------------------------------+
|result                                                                                                                 |result                                     |
+-------------------------------------------------------------------------------------------------------------+-----------------------------------------+
|[Which is the capital of France?, Who was the first president of USA?]|[Paris, George Washington]|
|[Which is the capital of Bulgaria ?]                                                              |[Sofia]                                     |
|[Who is Donald Trump?]                                                                                |[a United States citizen]      |
+------------------------------------------------------------------------------------------------------------+------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|google_t5_small_ssm_nq|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[t5]|
|Language:|en|
|Size:|179.1 MB|

## References

C4, Wikipedia, NQ