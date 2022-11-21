---
layout: model
title: T5 Question Generation (Small)
author: John Snow Labs
name: t5_question_generation_small
date: 2022-07-05
tags: [en, open_source]
task: Text Generation
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
recommended: true
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Text Generation model, originally trained on SQUAD dataset, then finetuned by AllenAI team, to generate questions from texts. The power lies on the ability to generate also questions providing a low number of tokens, for example a subject and a verb (`Amazon` `should provide`), what would return a question similar to `What Amazon should provide?`).

At the same time, this model can be used to feed Question Answering Models, as the first parameter (question), while providing a bigger paragraph as context. This way, you:
- First, generate questions on the fly
- Second, look for an answer in the text.

Moreover, the input of this model can even be a concatenation of entities from NER (`EMV` - ORG , `will provide` - ACTION).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_question_generation_small_en_4.0.0_3.0_1657032292222.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

t5 = T5Transformer() \
    .pretrained("t5_question_generation_small") \
    .setTask("")\
    .setMaxOutputLength(200)\
    .setInputCols(["documents"]) \
    .setOutputCol("question")

data_df = spark.createDataFrame([["EMV will pay"]]).toDF("text")

pipeline = Pipeline().setStages([document_assembler, t5])
results = pipeline.fit(data_df).transform(data_df)

results.select("question.result").show(truncate=False)
```
```scala
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

val t5 = T5Transformer.pretrained("t5_question_generation_small")
  .setTask("")
  .setMaxOutputLength(200)
  .setInputCols("documents")
  .setOutputCol("question")

val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

val data = Seq("EMV will pay").toDF("text")

val result = pipeline.fit(data).transform(data)

result.select("question.result").show(false)
```
</div>

## Results

```bash
+--------------------+
|result              |
+--------------------+
|[What will EMV pay?]|
+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_question_generation_small|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[summaries]|
|Language:|en|
|Size:|148.0 MB|

## References

SQUAD2.0
