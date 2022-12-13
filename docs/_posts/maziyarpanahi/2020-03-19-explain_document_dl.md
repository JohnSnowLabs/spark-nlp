---
layout: model
title: Explain Document DL
author: John Snow Labs
name: explain_document_dl
date: 2020-03-19
task: [Sentence Detection, Part of Speech Tagging, Lemmatization, Pipeline Public, Spell Check]
language: en
edition: Spark NLP 2.5.5
spark_version: 2.4
tags: [pipeline, en, open_source]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
The ``explain_document_dl`` is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/2da56c087da53a2fac1d51774d49939e05418e57/jupyter/annotation/english/explain-document-dl/Explain%20Document%20DL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_2.4.3_2.4_1584626657780.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_2.4.3_2.4_1584626657780.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
pipeline = PretrainedPipeline('explain_document_dl', lang = 'en')

annotations =  pipeline.fullAnnotate("""French author who helped pioner the science-fiction genre. Verne wrate about space, air, and underwater travel before navigable aircrast and practical submarines were invented, and before any means of space travel had been devised.""")[0]

annotations.keys()
```

```scala

val pipeline = new PretrainedPipeline('explain_document_dl', lang = 'en')

val result = pipeline.fullAnnotate("French author who helped pioner the science-fiction genre. Verne wrate about space, air, and underwater travel before navigable aircrast and practical submarines were invented, and before any means of space travel had been devised.")(0)
```

{:.nlu-block}
```python
import nlu

text = ["""John Snow built a detailed map of all the households where people died, and came to the conclusion that the fault was one public water pump that all the victims had used."""]
explain_df = nlu.load('en.explain.dl').predict(text)
explain_df
```

</div>

{:.h2_title}
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

{:.h2_title}
## Included Models 
The explain_document_dl has one Transformer and six annotators: 
- Documenssembler - A Transformer that creates a column that contains documents. 
- Sentence Segmenter - An annotator that produces the sentences of the document. 
- Tokenizer - An annotator that produces the tokens of the sentences. 
- SpellChecker - An annotator that produces the spelling-corrected tokens. 
- Stemmer - An annotator that produces the stems of the tokens. 
- Lemmatizer - An annotator that produces the lemmas of the tokens. 
- POS Tagger - An annotator that produces the parts of speech of the associated tokens.
