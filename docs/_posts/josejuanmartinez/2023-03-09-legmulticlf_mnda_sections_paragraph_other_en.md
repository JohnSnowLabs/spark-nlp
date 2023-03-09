---
layout: model
title: Multilabel Classification of NDA Clauses (paragraph, medium)
author: John Snow Labs
name: legmulticlf_mnda_sections_paragraph_other
date: 2023-03-09
tags: [nda, en, licensed, tensorflow]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: MultiClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This models is a version of `legmulticlf_mnda_sections_other` (sentence, medium) but expecting a bigger-than-sentence context, ideally between 2 and 4-5 sentences, or a small paragraph, to provide with more context.

It should be run on sentences of the NDA clauses, and will retrieve a series of 1..N labels for each of them. The possible clause types detected my this model in NDA / MNDA aggrements are:

1. Parties to the Agreement - Names of the Parties Clause  
2. Identification of What Information Is Confidential - Definition of Confidential Information Clause
3. Use of Confidential Information: Permitted Use Clause and Obligations of the Recipient
4. Time Frame of the Agreement - Termination Clause  
5. Return of Confidential Information Clause 
6. Remedies for Breaches of Agreement - Remedies Clause 
7. Non-Solicitation Clause
8. Dispute Resolution Clause  
9. Exceptions Clause  
10. Non-competition clause
11. Other: Nothing of the above (synonym to `[]`)-

## Predicted Entities

`APPLIC_LAW`, `ASSIGNMENT`, `DEF_OF_CONF_INFO`, `DISPUTE_RESOL`, `EXCEPTIONS`, `NAMES_OF_PARTIES`, `NON_COMP`, `NON_SOLIC`, `PREAMBLE`, `REMEDIES`, `REQ_DISCL`, `RETURN_OF_CONF_INFO`, `TERMINATION`, `USE_OF_CONF_INFO`, `OTHER`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legmulticlf_mnda_sections_paragraph_other_en_1.0.0_3.0_1678377832037.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legmulticlf_mnda_sections_paragraph_other_en_1.0.0_3.0_1678377832037.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = (
    nlp.DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
)

sentence_detector = (
    nlp.SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")
    .setExplodeSentences(True)
    .setCustomBounds(['\n'])
)


embeddings = (
    nlp.UniversalSentenceEncoder.pretrained()
    .setInputCols("document")
    .setOutputCol("sentence_embeddings")
)

paragraph_classifier = (
    nlp.MultiClassifierDLModel.pretrained("legmulticlf_mnda_sections_paragraph_other", "en", "legal/models")
    .setInputCols(["sentence_embeddings"])
    .setOutputCol("class")
)


sentence_pipeline = nlp.Pipeline(stages=[document_assembler, sentence_detector, embeddings, paragraph_classifier])
prediction_pipeline = nlp.Pipeline(stages=[document_assembler, embeddings, paragraph_classifier])

text = """RECITALS

WHEREAS, Corvus Gold Nevada Inc., a Nevada corporation (“Corvus”) and AngloGold Xxxxxxx (U.S.A.) Exploration, a Delaware corporation (“Viewer”) entered into that certain Confidentiality Agreement with an effective date of December 4, 2017 (“CA”); and
WHEREAS, Corvus and Viewer desire to amend the terms of the CA pursuant to the terms of this Amendment; and
WHEREAS, any terms not defined herein shall have the meanings set forth in the CA, as amended from time to time

EXECUTION VERSION
 
VITAL IMAGES, INC.
TOSHIBA MEDICAL SYSTEMS CORPORATION
 
Confidentiality Agreement
This Confidentiality Agreement (this “Agreement”) dated as of January 28, 2011, between VITAL IMAGES, INC., a Minnesota corporation (“Vital Images” or the “Company”), and TOSHIBA MEDICAL SYSTEMS CORPORATION, a Japanese corporation (“TMSC” or the “Receiving Company”).
W I T N E S S E T H:
WHEREAS, the Parties wish to consider a strategic business transaction (the “Transaction”) and, in connection therewith, desire to set forth certain agreements regarding such consideration and the sharing of confidential and proprietary information by Vital Images with TMSC;
"""

df = spark.createDataFrame([[""]]).toDF("text")

sentence_model = sentence_pipeline.fit(df)
prediction_model = prediction_pipeline.fit(df)

sentence_lp = nlp.LightPipeline(sentence_model)
prediction_lp = nlp.LightPipeline(prediction_model)

res = sentence_lp.fullAnnotate(text)
sentences = [x.result for x in res[0]['sentence']]

for i, s in enumerate(sentences):
    prev_sentence = "" if i==0 else sentences[i-1]
    next_sentence = "" if i>=len(sentences)-1 else sentences[i+1]
    chunk = " ".join([prev_sentence, s, next_sentence]).strip()
    print(f"{prediction_lp.annotate(chunk)['class']}: {chunk}")

```

</div>

## Results

```bash
['PREAMBLE']: WHEREAS, Corvus Gold Nevada Inc., a Nevada corporation (“Corvus”) and AngloGold Xxxxxxx (U.S.A.) Exploration, a Delaware corporation (“Viewer”) entered into that certain Confidentiality Agreement with an effective date of December 4, 2017 (“CA”); and WHEREAS, Corvus and Viewer desire to amend the terms of the CA pursuant to the terms of this Amendment;
['DEF_OF_CONF_INFO']: and WHEREAS, Corvus and Viewer desire to amend the terms of the CA pursuant to the terms of this Amendment; and
['OTHER', 'PREAMBLE']: WHEREAS, Corvus and Viewer desire to amend the terms of the CA pursuant to the terms of this Amendment; and WHEREAS, any terms not defined herein shall have the meanings set forth in the CA, as amended from time to time
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legmulticlf_mnda_sections_paragraph_other|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|13.4 MB|

## References

In-house MNDA

## Benchmarking

```bash
label precision    recall  f1-score   support
         APPLIC_LAW       0.86      0.84      0.85        57
         ASSIGNMENT       0.87      0.83      0.85        41
   DEF_OF_CONF_INFO       0.86      0.76      0.81        67
      DISPUTE_RESOL       0.84      0.69      0.76        70
         EXCEPTIONS       0.84      0.79      0.82       109
   NAMES_OF_PARTIES       0.90      0.76      0.83        50
           NON_COMP       0.79      0.67      0.72        33
          NON_SOLIC       0.81      0.82      0.81        82
              OTHER       0.91      0.89      0.90       838
           PREAMBLE       0.86      0.78      0.81        76
           REMEDIES       0.91      0.84      0.87        87
          REQ_DISCL       0.91      0.77      0.84        83
RETURN_OF_CONF_INFO       0.78      0.79      0.79        78
        TERMINATION       0.74      0.67      0.70        42
   USE_OF_CONF_INFO       0.77      0.84      0.80       200
          micro-avg       0.87      0.83      0.85      1913
          macro-avg       0.84      0.78      0.81      1913
       weighted-avg       0.87      0.83      0.85      1913
        samples-avg       0.82      0.84      0.83      1913
```