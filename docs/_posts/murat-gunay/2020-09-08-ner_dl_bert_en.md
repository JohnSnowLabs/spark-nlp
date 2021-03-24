---
layout: model
title: Detect Entities (BERT)
author: John Snow Labs
name: ner_dl_bert
date: 2020-09-08
task: Named Entity Recognition
language: en
edition: Spark NLP 2.6.0
tags: [ner, en, open_source]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
`ner_dl_bert` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together. `ner_dl_bert` model is trained with `bert_based_cased` word embeddings, so be sure to use the same embeddings in the pipeline.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_bert_en_2.6.0_2.4_1599550979101.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## Predicted Entities 
Persons-`PER`, Locations-`LOC`, Organizations-`ORG`, Miscellaneous-`MISC`.

## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}


```python
...
embeddings = BertEmbeddings.pretrained(name='bert_base_cased', lang='en') \
    .setInputCols(['document', 'token']) \
    .setOutputCol('embeddings')
ner_model = NerDLModel.pretrained("ner_dl_bert", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
...        
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, ner_model, ner_converter])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text'))

result = pipeline_model.transform(spark.createDataFrame(pd.DataFrame({'text': ["""William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being the largest individual shareholder until May 2014. He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico; it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect. During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000. He gradually transferred his duties to Ray Ozzie and Craig Mundie. He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."""]})))
```

```scala
...
val embeddings = BertEmbeddings.pretrained(name="bert_base_cased", lang="en")
    .setInputCols(Array('document', 'token'))
    .setOutputCol('embeddings')
val ner_model = NerDLModel.pretrained("ner_dl_bert", "en")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings, ner_model, ner_converter))

val result = pipeline.fit(Seq.empty["William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being the largest individual shareholder until May 2014. He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico; it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect. During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000. He gradually transferred his duties to Ray Ozzie and Craig Mundie. He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."].toDS.toDF("text")).transform(data)
```

{:.nlu-block}
```python
import nlu
text = ["""William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being the largest individual shareholder until May 2014. He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico; it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect. During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000. He gradually transferred his duties to Ray Ozzie and Craig Mundie. He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."""]

ner_df = nlu.load('en.ner.dl.bert').predict(text, output_level = "chunk")
ner_df[["entities", "entities_confidence"]]
```
</div>

{:.h2_title}
## Results

```bash
+-----------------------+---------+
|chunk                  |ner_label|
+-----------------------+---------+
|William Henry Gates III|PER      |
|American               |MISC     |
|Microsoft Corporation  |ORG      |
|Microsoft              |ORG      |
|Gates                  |PER      |
|Seattle                |LOC      |
|Washington             |LOC      |
|Gates                  |PER      |
|Microsoft              |ORG      |
|Paul Allen             |PER      |
|Albuquerque            |LOC      |
|New Mexico             |LOC      |
+-----------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_dl_bert|
|Type:|ner|
|Compatibility:| Spark NLP 2.6.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|false|


{:.h2_title}
## Data Source
The model is trained based on data from[CoNLL 2003 Data Set](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003)