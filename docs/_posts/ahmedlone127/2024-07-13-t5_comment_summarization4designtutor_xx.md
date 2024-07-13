---
layout: model
title: Multilingual T5ForConditionalGeneration Cased model (from qiaoyi)
author: John Snow Labs
name: t5_comment_summarization4designtutor
date: 2024-07-13
tags: [ro, fr, de, en, open_source, t5, xx, onnx]
task: Text Generation
language: xx
edition: Spark NLP 5.4.1
spark_version: 3.0
supported: true
engine: onnx
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Comment_Summarization4DesignTutor` is a Multilingual model originally trained by `qiaoyi`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_comment_summarization4designtutor_xx_5.4.1_3.0_1720883007023.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_comment_summarization4designtutor_xx_5.4.1_3.0_1720883007023.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

t5 = T5Transformer.pretrained("t5_comment_summarization4designtutor","xx") \
    .setInputCols(["document"]) \
    .setOutputCol("answers")

pipeline = Pipeline(stages=[documentAssembler, t5])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
      .setInputCols("text")
      .setOutputCols("document")

val t5 = T5Transformer.pretrained("t5_comment_summarization4designtutor","xx")
    .setInputCols("document")
    .setOutputCol("answers")

val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_comment_summarization4designtutor|
|Compatibility:|Spark NLP 5.4.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|xx|
|Size:|324.5 MB|

## References

References

- https://huggingface.co/qiaoyi/Comment_Summarization4DesignTutor
- https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html
- https://arxiv.org/abs/1805.12471
- https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf
- https://aclanthology.org/I05-5002
- https://arxiv.org/abs/1708.00055
- https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs
- https://arxiv.org/abs/1704.05426
- https://arxiv.org/abs/1606.05250
- https://link.springer.com/chapter/10.1007/11736790_9
- https://semanticsarchive.net/Archive/Tg3ZGI2M/Marneffe.pdf
- https://www.researchgate.net/publication/221251392_Choice_of_Plausible_Alternatives_An_Evaluation_of_Commonsense_Causal_Reasoning
- https://arxiv.org/abs/1808.09121
- https://aclanthology.org/N18-1023
- https://arxiv.org/abs/1810.12885
- https://arxiv.org/abs/1905.10044
- https://arxiv.org/pdf/1910.10683.pdf
- https://camo.githubusercontent.com/623b4dea0b653f2ad3f36c71ebfe749a677ac0a1/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f6d61782f343030362f312a44304a31674e51663876727255704b657944387750412e706e67