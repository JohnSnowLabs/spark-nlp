---
layout: model
title: Classifying Proposal Comments
author: John Snow Labs
name: legclf_bert_support_proposal
date: 2023-02-17
tags: [en, legal, licensed, classification, proposal, support, tensorflow]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: LegalBertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Given a proposal on a socially important issue, the model classifies whether a comment is `In_Favor`, `Against`, or `Neutral` towards the proposal.

## Predicted Entities

`In_Favor`, `Neutral`, `Against`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_bert_support_proposal_en_1.0.0_3.0_1676599695968.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_bert_support_proposal_en_1.0.0_3.0_1676599695968.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["document"]) \
    .setOutputCol("token")

classifier = legal.BertForSequenceClassification.pretrained("legclf_bert_support_proposal", "en", "legal/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("class")

clf_pipeline = nlp.Pipeline(stages=[
    document_assembler, 
    tokenizer,
    classifier    
])

empty_df = spark.createDataFrame([['']]).toDF("text")

model = clf_pipeline.fit(empty_df)

sample_text = ["""This is one of the most boring movies I have ever seen, its horrible. Christopher Lee is good but he is hardly in it, the only good part is the opening scene. Don't be fooled by the title. "End of the World" is truly a bad movie, I stopped watching it close to the end it was so bad, only for die-hard b-movie fans that have the brain to stand this vomit.""",

"""Of course, there is still a lot of possible improvement in the pipeline, but we definitely don't have to wait for some genius new technology to start. Why am I so definitely against this proposal though it sounds so reasonable and helpful? I'm definitely against the notion that we'll have to wait for a new genius industrial technology to show up to even think of starting a proper transformation. In my opinion, the opposite is true: We have to start right now with what we have & by the way develop better concepts of how to use all the technology & methods already available optimally. And for me, nuclear energy which is - by the way - relaunched with this proposal, is definitely not part of the game, not even in the modular mini-nuke version of Mr. Gates. There are people who know much more about renewable energy than Mr. Gates & completely energy independent who hate that book because of this crap.""",

"""One common defense policy would strengthen the voice and influence in our own backyard. A strong EU army can be a stabilizing factor in the unstable regions around our continent. We Europeans should take our safety and defense into our own hands and not rely on the US to do it for us."""
]

test = spark.createDataFrame(pd.DataFrame({"text": sample_text}))

result = model.transform(test)
```

</div>

## Results

```bash
+--------+--------------------+
|   class|            document|
+--------+--------------------+
| Neutral|This is one of the...|
| Against|Of course, there ...|
|In_Favor|One common defense...|
+--------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_bert_support_proposal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|403.0 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Train dataset available [here](https://touche.webis.de/clef23/touche23-web/multilingual-stance-classification.html#data)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
Against       0.84       0.87    0.86      85      
In_Favor      0.87       0.84    0.86      90      
Neutral       0.98       0.98    0.98      57      
accuracy      -          -       0.89      232     
macro-avg     0.90       0.90    0.90      232     
weighted-avg  0.89       0.89    0.89      232   
```
