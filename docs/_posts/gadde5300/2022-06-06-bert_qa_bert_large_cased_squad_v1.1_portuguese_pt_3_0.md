---
layout: model
title: Portuguese BertForQuestionAnswering model (from pierreguillou)
author: John Snow Labs
name: bert_qa_bert_large_cased_squad_v1.1_portuguese
date: 2022-06-06
tags: [pt, open_source, question_answering, bert]
task: Question Answering
language: pt
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-large-cased-squad-v1.1-portuguese` is a Portuguese model orginally trained by `pierreguillou`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_qa_bert_large_cased_squad_v1.1_portuguese_pt_4.0.0_3.0_1654536169488.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = MultiDocumentAssembler() \ 
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = BertForQuestionAnswering.pretrained("bert_qa_bert_large_cased_squad_v1.1_portuguese","pt") \
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
.pretrained("bert_qa_bert_large_cased_squad_v1.1_portuguese","pt")
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
nlu.load("pt.answer_question.squad.bert.large_cased").predict("""What's my name?|||"My name is Clara and I live in Berkeley.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_qa_bert_large_cased_squad_v1.1_portuguese|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|pt|
|Size:|1.2 GB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/pierreguillou/bert-large-cased-squad-v1.1-portuguese
- https://github.com/piegu/language-models/blob/master/question_answering_BERT_large_cased_squad_v11_pt.ipynb
- https://nbviewer.jupyter.org/github/piegu/language-models/blob/master/question_answering_BERT_large_cased_squad_v11_pt.ipynb
- https://medium.com/@pierre_guillou/nlp-como-treinar-um-modelo-de-question-answering-em-qualquer-linguagem-baseado-no-bert-large-1c899262dd96#c2f5
- https://ailab.unb.br/
- https://www.linkedin.com/in/pierreguillou/
- http://www.deeplearningbrasil.com.br/
- https://neuralmind.ai/
- https://medium.com/@pierre_guillou/nlp-como-treinar-um-modelo-de-question-answering-em-qualquer-linguagem-baseado-no-bert-large-1c899262dd96