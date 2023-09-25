---
layout: model
title: Atgenomix Testing QA Model
author: LIN-Yu-Ting
name: AtgxRobertaBaseSquad2
date: 2023-09-18
tags: [en, open_source, tensorflow]
task: Question Answering
language: en
edition: Spark NLP 4.4.3
spark_version: 3.4
supported: false
engine: tensorflow
annotator: RoBertaForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Testing Question Answering model for Atgenomix usage

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/LIN-Yu-Ting/AtgxRobertaBaseSquad2_en_4.4.3_3.4_1695000774804.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://community.johnsnowlabs.com/LIN-Yu-Ting/AtgxRobertaBaseSquad2_en_4.4.3_3.4_1695000774804.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
spark = sparknlp.start()
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|AtgxRobertaBaseSquad2|
|Compatibility:|Spark NLP 4.4.3+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|460.0 MB|
|Case sensitive:|true|
|Max sentence length:|512|