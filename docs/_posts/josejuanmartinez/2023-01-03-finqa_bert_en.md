---
layout: model
title: Financial Question Answering (Bert)
author: John Snow Labs
name: finqa_bert
date: 2023-01-03
tags: [en, licensed]
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

Financial Bert-based Question Answering model, trained on squad-v2, finetuned on proprietary Financial questions and answers.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finqa_bert_en_1.0.0_3.0_1672759463237.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.MultiDocumentAssembler()\
        .setInputCols(["question", "context"])\
        .setOutputCols(["document_question", "document_context"])

spanClassifier = nlp.BertForQuestionAnswering.pretrained("finqa_bert","en", "finance/models") \
       .setInputCols(["document_question", "document_context"]) \
       .setOutputCol("answer") \
       .setCaseSensitive(True)

pipeline = Pipeline().setStages([
        documentAssembler,
        spanClassifier
])

example = spark.createDataFrame([["On which market is their common stock traded?", "Our common stock is traded on the Nasdaq Global Select Market under the symbol CDNS."]]).toDF("question", "context")

result = pipeline.fit(example).transform(example)

result.select('answer.result').show()
```

</div>

## Results

```bash
`Nasdaq Global Select Market under the symbol CDNS`
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finqa_bert|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|407.9 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Trained on squad-v2, finetuned on proprietary Financial questions and answers.