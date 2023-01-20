---
layout: model
title: Financial Question Answering (RoBerta)
author: John Snow Labs
name: finqa_roberta
date: 2022-08-09
tags: [en, finance, qa, licensed]
task: Question Answering
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Financial RoBerta-based Question Answering model, trained on squad-v2, finetuned on proprietary Financial questions and answers.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finqa_roberta_en_1.0.0_3.2_1660054527812.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finqa_roberta_en_1.0.0_3.2_1660054527812.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python

documentAssembler = nlp.MultiDocumentAssembler()\
        .setInputCols(["question", "context"])\
        .setOutputCols(["document_question", "document_context"])

spanClassifier = nlp.RoBertaForQuestionAnswering.pretrained("finqa_roberta","en", "finance/models") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer") \
.setCaseSensitive(True)


pipeline = Pipeline(stages=[documentAssembler, spanClassifier])

example = spark.createDataFrame([["What is the current total Operating Profit?", "Operating profit totaled EUR 9.4 mn , down from EUR 11.7 mn in 2004"]]).toDF("question", "context")

result = pipeline.fit(example).transform(example)

result.select('answer.result').show()
```

</div>

## Results

```bash
`9.4 mn , down from EUR 11.7`
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finqa_roberta|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[question, context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|248.1 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Trained on squad-v2, finetuned on proprietary Financial questions and answers.
