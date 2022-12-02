---
layout: model
title: Spanish RobertaForQuestionAnswering (from PlanTL-GOB-ES)
author: John Snow Labs
name: roberta_qa_PlanTL_GOB_ES_roberta_base_bne_sqac
date: 2022-06-20
tags: [es, open_source, question_answering, roberta]
task: Question Answering
language: es
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: RoBertaForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-base-bne-sqac` is a Spanish model originally trained by `PlanTL-GOB-ES`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_qa_PlanTL_GOB_ES_roberta_base_bne_sqac_es_4.0.0_3.0_1655730149548.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = MultiDocumentAssembler() \ 
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = RoBertaForQuestionAnswering.pretrained("roberta_qa_PlanTL_GOB_ES_roberta_base_bne_sqac","es") \
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
.pretrained("roberta_qa_PlanTL_GOB_ES_roberta_base_bne_sqac","es")
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
nlu.load("es.answer_question.sqac.roberta.base").predict("""What's my name?|||"My name is Clara and I live in Berkeley.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_qa_PlanTL_GOB_ES_roberta_base_bne_sqac|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[question, context]|
|Output Labels:|[answer]|
|Language:|es|
|Size:|460.3 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-sqac
- https://arxiv.org/abs/2107.07253
- http://www.bne.es/en/Inicio/index.html
- https://arxiv.org/abs/1907.11692
- https://github.com/PlanTL-GOB-ES/lm-spanish