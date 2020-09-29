---
layout: model
title: 
author: John Snow Labs
name: 
date: 
tags: [pipeline, en]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
Short description of the pipeline and its use.

{:.btn-box}
[Live Demo](){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/explain-document-ml/explain_document_ml.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_2.4.3_2.4_1584626657780.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala

code example
```

```python

pipeline = PretrainedPipeline('explain_document_dl', lang =' en').annotate(' Hello world!')
```

</div>

## Results
Add the results returned by the above code...

{:.result_box}
```bash
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|                text|            document|            sentence|               token|               spell|              lemmas|               stems|                 pos|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|French author who...|[[document, 0, 23...|[[document, 0, 57...|[[token, 0, 5, Fr...|[[token, 0, 5, Fr...|[[token, 0, 5, Fr...|[[token, 0, 5, fr...|[[pos, 0, 5, JJ, ...|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
```

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Pipeline Name:|Name of the model - same as in the metadata (header of this file)|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.5.5+|
|License:|Open Source or Licensed|
|Edition:|Official or Community|
|Language:|en|


## Included Models 
The list of models included in the pipeline
