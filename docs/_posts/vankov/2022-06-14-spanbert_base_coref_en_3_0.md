---
layout: model
title: SpanBert Coreference Resolution
author: John Snow Labs
name: spanbert_base_coref
date: 2022-06-14
tags: [en, open_source]
task: Dependency Parser
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: SpanBertCorefModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A coreference resolution model identifies expressions which refer to the same entity in a text. For example, given a sentence "John told Mary he would like to borrow a book from her." the model will link "he" to "John" and "her" to "Mary".  This model is based on SpanBert, which is fine-tuned on the OntoNotes 5.0 data set.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spanbert_base_coref_en_4.0.0_3.0_1655203982784.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
data = spark.createDataFrame([["John told Mary he would like to borrow a book from her."]]).toDF("text")
document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentences")
tokenizer = Tokenizer().setInputCols(["sentences"]).setOutputCol("tokens")
corefResolution = SpanBertCorefModel().pretrained("spanbert_base_coref").setInputCols(["sentences", "tokens"]).setOutputCol("corefs")
pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, corefResolution])

model = pipeline.fit(self.data)

model.transform(self.data).selectExpr("explode(corefs) AS coref").selectExpr("coref.result as token", "coref.metadata").show(truncate=False)
```
```scala
val data = Seq("John told Mary he would like to borrow a book from her.").toDF("text")
val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentencer = SentenceDetector().setInputCols(Array("document")).setOutputCol("sentences")
val tokenizer = new Tokenizer().setInputCols(Array("sentences")).setOutputCol("tokens")
val corefResolution = SpanBertCorefModel.pretrained("spanbert_base_coref").setInputCols(Array("sentences", "tokens")).setOutputCol("corefs")

val pipeline = new Pipeline().setStages(Array(document, sentencer, tokenizer, corefResolution))

val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(corefs) as coref").selectExpr("coref.result as token", "coref.metadata").show(truncate = false)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.coreference.spanbert").predict("""John told Mary he would like to borrow a book from her.""")
```

</div>

## Results

```bash
+-----+------------------------------------------------------------------------------------+
|token|metadata                                                                            |
+-----+------------------------------------------------------------------------------------+
|John |{head.sentence -> -1, head -> ROOT, head.begin -> -1, head.end -> -1, sentence -> 0}|
|he   |{head.sentence -> 0, head -> John, head.begin -> 0, head.end -> 3, sentence -> 0}   |
|Mary |{head.sentence -> -1, head -> ROOT, head.begin -> -1, head.end -> -1, sentence -> 0}|
|her  |{head.sentence -> 0, head -> Mary, head.begin -> 10, head.end -> 13, sentence -> 0} |
+-----+------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|spanbert_base_coref|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentences, tokens]|
|Output Labels:|[corefs]|
|Language:|en|
|Size:|566.3 MB|
|Case sensitive:|true|

## References

OntoNotes 5.0

## Benchmarking

```bash
label score
f1  77.7
```
https://github.com/mandarjoshi90/coref
