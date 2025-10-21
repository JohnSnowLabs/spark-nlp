{%- capture title -%}
DeBertaForZeroShotClassification
{%- endcapture -%}

{%- capture description -%}
The DeBERTa model was proposed in [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://huggingface.co/papers/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen It is based on Google’s BERT model released in 2018 and Facebook’s RoBERTa model released in 2019.

It builds on RoBERTa with disentangled attention and enhanced mask decoder training with half of the data used in RoBERTa.

Pretrained models can be loaded with `pretrained` of the companion object:
```scala
val zeroShotClassifier = DeBertaForZeroShotClassification.pretrained()
    .setInputCols("document", "token")
    .setOutputCol("class")
    .setCandidateLabels(Array("urgent", "mobile", "travel", "movie", "music", "sport", "weather", "technology"))
```
The default model is `"deberta_base_zero_shot_classifier_mnli_anli_v3"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=DeBertaForZeroShotClassification).

Spark NLP also supports Hugging Face transformer-based code generation models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Sources** :

- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://huggingface.co/papers/2006.03654)  
- [DeBERTa on GitHub](https://github.com/microsoft/DeBERTa)   
- [DeBERTa on SuperGLUE Leaderboard](https://super.gluebenchmark.com/leaderboard)

**Paper abstract**

*Recent progress in pre-trained neural language models has significantly improved the performance of many natural language processing (NLP) tasks. In this paper we propose a new model architecture DeBERTa (Decoding-enhanced BERT with disentangled attention) that improves the BERT and RoBERTa models using two novel techniques. The first is the disentangled attention mechanism, where each word is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices on their contents and relative positions, respectively. Second, an enhanced mask decoder is used to incorporate absolute positions in the decoding layer to predict the masked tokens in model pre-training. In addition, a new virtual adversarial training method is used for fine-tuning to improve models' generalization. We show that these techniques significantly improve the efficiency of model pre-training and the performance of both natural language understanding (NLU) and natural langauge generation (NLG) downstream tasks. Compared to RoBERTa-Large, a DeBERTa model trained on half of the training data performs consistently better on a wide range of NLP tasks, achieving improvements on MNLI by +0.9% (90.2% vs. 91.1%), on SQuAD v2.0 by +2.3% (88.4% vs. 90.7%) and RACE by +3.6% (83.2% vs. 86.8%). Notably, we scale up DeBERTa by training a larger version that consists of 48 Transform layers with 1.5 billion parameters. The significant performance boost makes the single DeBERTa model surpass the human performance on the SuperGLUE benchmark (Wang et al., 2019a) for the first time in terms of macro-average score (89.9 versus 89.8), and the ensemble DeBERTa model sits atop the SuperGLUE leaderboard as of January 6, 2021, out performing the human baseline by a decent margin (90.3 versus 89.8).*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
CLASS
{%- endcapture -%}

{%- capture api_link -%}
[DeBertaForZeroShotClassification](/api/com/johnsnowlabs/nlp/annotators/classifier/dl/DeBertaForZeroShotClassification.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[DeBertaForZeroShotClassification](/api/python/reference/autosummary/sparknlp/annotator/classifier_dl/deberta_for_zero_shot_classification/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[DeBertaForZeroShotClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/DeBertaForZeroShotClassification.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, DeBertaForZeroShotClassification
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

zero_shot_classifier = DeBertaForZeroShotClassification.pretrained() \
    .setInputCols(['token', 'document']) \
    .setOutputCol('class') \
    .setCaseSensitive(True) \
    .setMaxSentenceLength(512) \
    .setCandidateLabels(["urgent", "mobile", "travel", "movie", "music", "sport", "weather", "technology"])

pipeline = Pipeline().setStages([
    document_assembler,
    tokenizer,
    zero_shot_classifier
])

text_data = [
    ["I have a problem with my iphone that needs to be resolved asap!!"],
    ["Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app."],
    ["I have a phone and I love it!"],
    ["I really want to visit Germany and I am planning to go there next year."],
    ["Let's watch some movies tonight! I am in the mood for a horror movie."],
    ["Have you watched the match yesterday? It was a great game!"],
    ["We need to harry up and get to the airport. We are going to miss our flight!"]
]

input_df = spark.createDataFrame(text_data, ["text"])

model = pipeline.fit(input_df)
result = model.transform(input_df)

result.select("class.result").show(truncate=False)

+---------+
|result   |
+---------+
|[music]  |
|[weather]|
|[sport]  |
|[sport]  |
|[music]  |
|[sport]  |
|[weather]|
+---------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.classifier.dl.DeBertaForZeroShotClassification
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val zeroShotClassifier = DeBertaForZeroShotClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("class")
  .setCaseSensitive(true)
  .setMaxSentenceLength(512)
  .setCandidateLabels(Array(
    "urgent", "mobile", "travel", "movie", "music", "sport", "weather", "technology"
  ))

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  zeroShotClassifier
))

val textData = Seq(
  "I have a problem with my iphone that needs to be resolved asap!!",
  "Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app.",
  "I have a phone and I love it!",
  "I really want to visit Germany and I am planning to go there next year.",
  "Let's watch some movies tonight! I am in the mood for a horror movie.",
  "Have you watched the match yesterday? It was a great game!",
  "We need to harry up and get to the airport. We are going to miss our flight!"
).toDF("text")

val model = pipeline.fit(textData)
val result = model.transform(textData)

result.select("class.result").show(false)

+---------+
|result   |
+---------+
|[music]  |
|[weather]|
|[sport]  |
|[sport]  |
|[music]  |
|[sport]  |
|[weather]|
+---------+
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}