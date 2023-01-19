---
layout: model
title: English XlmRoBertaForQuestionAnswering (from deepset)
author: John Snow Labs
name: xlm_roberta_qa_xlm_roberta_large_squad2
date: 2022-06-23
tags: [en, open_source, question_answering, xlmroberta]
task: Question Answering
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `xlm-roberta-large-squad2` is a English model originally trained by `deepset`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_qa_xlm_roberta_large_squad2_en_4.0.0_3.0_1655996042440.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_qa_xlm_roberta_large_squad2_en_4.0.0_3.0_1655996042440.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = MultiDocumentAssembler() \ 
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = XlmRoBertaForQuestionAnswering.pretrained("xlm_roberta_qa_xlm_roberta_large_squad2","en") \
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
.setInputCols(Array("question", "context")) 
.setOutputCols(Array("document_question", "document_context"))

val spanClassifier = XlmRoBertaForQuestionAnswering
.pretrained("xlm_roberta_qa_xlm_roberta_large_squad2","en")
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
nlu.load("en.answer_question.squadv2.xlm_roberta.large").predict("""What's my name?|||"My name is Clara and I live in Berkeley.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_qa_xlm_roberta_large_squad2|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[question, context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|1.9 GB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/deepset/xlm-roberta-large-squad2
- https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
- https://www.linkedin.com/company/deepset-ai/
- https://twitter.com/deepset_ai
- http://www.deepset.ai/jobs
- https://haystack.deepset.ai/community/join
- https://public-mlflow.deepset.ai/#/experiments/124/runs/3a540e3f3ecf4dd98eae8fc6d457ff20
- https://github.com/deepset-ai/haystack/
- https://github.com/deepset-ai/FARM
- https://deepset.ai/germanquad
- https://github.com/deepmind/xquad
- https://deepset.ai
- https://deepset.ai/german-bert
- https://github.com/deepset-ai/haystack/discussions
- https://github.com/facebookresearch/MLQA