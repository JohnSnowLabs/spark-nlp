---
layout: model
title: English RobertaForQuestionAnswering (from anas-awadalla)
author: John Snow Labs
name: roberta_qa_roberta_base_few_shot_k_32_finetuned_squad_seed_6
date: 2022-06-20
tags: [en, open_source, question_answering, roberta]
task: Question Answering
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: RoBertaForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-base-few-shot-k-32-finetuned-squad-seed-6` is a English model originally trained by `anas-awadalla`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_qa_roberta_base_few_shot_k_32_finetuned_squad_seed_6_en_4.0.0_3.0_1655732701264.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = MultiDocumentAssembler() \ 
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = RoBertaForQuestionAnswering.pretrained("roberta_qa_roberta_base_few_shot_k_32_finetuned_squad_seed_6","en") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer") \
.setCaseSensitive(True)

pipeline = Pipeline().setStages([
document_assembler,
spanClassifier
])

example = spark.createDataFrame([["What's my name?", "My name is Clara and I live in Berkeley."]]).toDF("question", "context")

result = pipeline.fit(example).transform(example)
```
```scala
val document = new MultiDocumentAssembler()
.setInputCols("question", "context")
.setOutputCols("document_question", "document_context")

val spanClassifier = RoBertaForQuestionAnswering
.pretrained("roberta_qa_roberta_base_few_shot_k_32_finetuned_squad_seed_6","en")
.setInputCols(Array("document_question", "document_context"))
.setOutputCol("answer")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document, spanClassifier))

val example = Seq(
("Where was John Lenon born?", "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London."),
("What's my name?", "My name is Clara and I live in Berkeley."))
.toDF("question", "context")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.answer_question.squad.roberta.base_32d_seed_6").predict("""What's my name?|||"My name is Clara and I live in Berkeley.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_qa_roberta_base_few_shot_k_32_finetuned_squad_seed_6|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[question, context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|417.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/anas-awadalla/roberta-base-few-shot-k-32-finetuned-squad-seed-6