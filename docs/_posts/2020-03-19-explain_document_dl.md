---
layout: model
title: Explain Document DL
author: John Snow Labs
name: explain_document_dl
date: 2020-03-19 00:00:00 +0800
tags: [pipeline, en, open_source]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
The *explain_document_dl* is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/explain-document-ml/explain_document_ml.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_2.4.3_2.4_1584626657780.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala

```

```python

pipeline = PretrainedPipeline('explain_document_dl', lang =' en').annotate(' Hello world!')
```

</div>
## Results

{:.result_box}
```bash
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|                text|            document|            sentence|               token|               spell|              lemmas|               stems|                 pos|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
|French author who...|[[document, 0, 23...|[[document, 0, 57...|[[token, 0, 5, Fr...|[[token, 0, 5, Fr...|[[token, 0, 5, Fr...|[[token, 0, 5, fr...|[[pos, 0, 5, JJ, ...|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_document_dl|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.5.5+|
|License:|Open Source|
|Edition:|Community|
|Language:|[en]|


## Included Models 
The explain_document_ml has one Transformer and six annotators: 
- Documenssembler - A Transformer that creates a column that contains documents. 
- Sentence Segmenter - An annotator that produces the sentences of the document. 
- Tokenizer - An annotator that produces the tokens of the sentences. 
- SpellChecker - An annotator that produces the spelling-corrected tokens. 
- Stemmer - An annotator that produces the stems of the tokens. 
- Lemmatizer - An annotator that produces the lemmas of the tokens. 
- POS Tagger - An annotator that produces the parts of speech of the associated tokens.

