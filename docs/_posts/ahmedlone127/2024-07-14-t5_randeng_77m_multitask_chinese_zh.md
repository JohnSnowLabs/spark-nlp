---
layout: model
title: Chinese T5ForConditionalGeneration Cased model (from IDEA-CCNL)
author: John Snow Labs
name: t5_randeng_77m_multitask_chinese
date: 2024-07-14
tags: [zh, open_source, t5, onnx]
task: Text Generation
language: zh
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

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Randeng-T5-77M-MultiTask-Chinese` is a Chinese model originally trained by `IDEA-CCNL`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_randeng_77m_multitask_chinese_zh_5.4.1_3.0_1720962928103.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_randeng_77m_multitask_chinese_zh_5.4.1_3.0_1720962928103.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_randeng_77m_multitask_chinese","zh") \
    .setInputCols("document") \
    .setOutputCol("answers")
    
pipeline = Pipeline(stages=[documentAssembler, t5])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols("text")
      .setOutputCols("document")
       
val t5 = T5Transformer.pretrained("t5_randeng_77m_multitask_chinese","zh") 
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
|Model Name:|t5_randeng_77m_multitask_chinese|
|Compatibility:|Spark NLP 5.4.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|zh|
|Size:|349.1 MB|

## References

References

- https://huggingface.co/IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese
- https://github.com/IDEA-CCNL/Fengshenbang-LM
- https://fengshenbang-doc.readthedocs.io/
- http://jmlr.org/papers/v21/20-074.html
- https://github.com/IDEA-CCNL/Fengshenbang-LM/
- https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/pretrain_t5
- https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/mt5_summary
- https://github.com/IDEA-CCNL/Fengshenbang-LM/
- https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/pretrain_t5
- https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/mt5_summary
- https://arxiv.org/abs/2209.02970
- https://arxiv.org/abs/2209.02970
- https://github.com/IDEA-CCNL/Fengshenbang-LM/
- https://github.com/IDEA-CCNL/Fengshenbang-LM/