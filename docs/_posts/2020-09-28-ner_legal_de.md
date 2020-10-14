---
layout: model
title: Detect legal entities in German
author: John Snow Labs
name: ner_legal
date: 2020-09-28
tags: [ner, de, licensed]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model can be used to detect legal entities in German text.
## Predicted Entities
STR, LIT, PER, EUN, VT, MRK, INN, UN, RS, ORG, GS, VS, LDS, GRT, VO, RR, LD, AN, ST

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_LEGAL_DE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_LEGAL_DE.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_legal_de_2.5.5_2.4_1599471454959.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python

clinical_ner = NerDLModel.pretrained("ner_legal", "de", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate("Dementsprechend hat der Bundesgerichtshof mit Beschluss vom 24 August 2017 ( - III ZA 15/17 - ) das bei ihm von der Antragstellerin anhängig gemachte „ Prozesskostenhilfeprüfungsverfahre“ an das Bundesarbeitsgericht abgegeben.")

```

</div>

{:.h2_title}
## Results

```bash
+----+-----------------------------------------------------+---------+---------+------------+
|    | chunk                                               |   begin |   end   | entity     |
+====+=====================================================+=========+=========+============+
|  0 | Bundesgerichtshof                                   |    24   |    40   | GRT        |
+----+-----------------------------------------------------+---------+---------+------------+
|  1 | Beschluss vom 24 August 2017 ( - III ZA 15/17 - )   |    46   |    94   | RS         |
+----+-----------------------------------------------------+---------+---------+------------+
|  2 | Bundesarbeitsgericht                                |    195  |   214   | GRT        |
+----+-----------------------------------------------------+---------+---------+------------+
```
{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_legal|
|Type:|ner|
|Compatibility:|Spark NLP for Healthcare 2.6.0 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|[de]|
|Case sensitive:|false|

