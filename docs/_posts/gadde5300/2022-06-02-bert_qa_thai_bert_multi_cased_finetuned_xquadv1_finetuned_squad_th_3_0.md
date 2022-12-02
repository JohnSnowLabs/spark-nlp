---
layout: model
title: Thai BertForQuestionAnswering model (from wicharnkeisei)
author: John Snow Labs
name: bert_qa_thai_bert_multi_cased_finetuned_xquadv1_finetuned_squad
date: 2022-06-02
tags: [th, open_source, question_answering, bert]
task: Question Answering
language: th
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `thai-bert-multi-cased-finetuned-xquadv1-finetuned-squad` is a Thai model orginally trained by `wicharnkeisei`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_qa_thai_bert_multi_cased_finetuned_xquadv1_finetuned_squad_th_4.0.0_3.0_1654192425473.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = MultiDocumentAssembler() \ 
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = BertForQuestionAnswering.pretrained("bert_qa_thai_bert_multi_cased_finetuned_xquadv1_finetuned_squad","th") \
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

val spanClassifier = BertForQuestionAnswering
.pretrained("bert_qa_thai_bert_multi_cased_finetuned_xquadv1_finetuned_squad","th")
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
nlu.load("th.answer_question.xquad_squad.bert.cased").predict("""What's my name?|||"My name is Clara and I live in Berkeley.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_qa_thai_bert_multi_cased_finetuned_xquadv1_finetuned_squad|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|th|
|Size:|665.6 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/wicharnkeisei/thai-bert-multi-cased-finetuned-xquadv1-finetuned-squad
- https://github.com/iapp-technology/iapp-wiki-qa-dataset