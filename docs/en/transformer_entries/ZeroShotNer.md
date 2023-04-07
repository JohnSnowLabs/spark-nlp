{%- capture title -%}
ZeroShotNer
{%- endcapture -%}

{%- capture description -%}
ZeroShotNerModel implements zero shot named entity recognition by utilizing RoBERTa
transformer models fine tuned on a question answering task.

Its input is a list of document annotations and it automatically generates questions which are
used to recognize entities. The definitions of entities is given by a dictionary structures,
specifying a set of questions for each entity. The model is based on
RoBertaForQuestionAnswering.

For more extended examples see the
[Examples](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/named-entity-recognition/ZeroShot_NER.ipynb).

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val zeroShotNer = ZeroShotNerModel.pretrained()
  .setInputCols("document")
  .setOutputCol("zer_shot_ner")
```

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Zero-Shot-NER).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture python_example -%}
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")
tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")
zero_shot_ner = ZeroShotNerModel() \
    .pretrained() \
    .setEntityDefinitions(
        {
            "NAME": ["What is his name?", "What is my name?", "What is her name?"],
            "CITY": ["Which city?", "Which is the city?"]
        }) \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("zero_shot_ner") \
data = spark.createDataFrame(
        [["My name is Clara, I live in New York and Hellen lives in Paris."]]
    ).toDF("text")
Pipeline() \
    .setStages([document_assembler, sentence_detector, tokenizer, zero_shot_ner]) \
    .fit(data) \
    .transform(data) \
    .selectExpr("document", "explode(zero_shot_ner) AS entity") \
    .select(
        "document.result",
        "entity.result",
        "entity.metadata.word",
        "entity.metadata.confidence",
        "entity.metadata.question") \
    .show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
val documentAssembler = new DocumentAssembler()
   .setInputCol("text")
   .setOutputCol("document")

 val sentenceDetector = new SentenceDetector()
   .setInputCols(Array("document"))
   .setOutputCol("sentences")

 val zeroShotNer = ZeroShotNerModel
   .pretrained()
   .setEntityDefinitions(
     Map(
       "NAME" -> Array("What is his name?", "What is her name?"),
       "CITY" -> Array("Which city?")))
   .setPredictionThreshold(0.01f)
   .setInputCols("sentences")
   .setOutputCol("zero_shot_ner")

 val pipeline = new Pipeline()
   .setStages(Array(
     documentAssembler,
     sentenceDetector,
     zeroShotNer))

 val model = pipeline.fit(Seq("").toDS.toDF("text"))
 val results = model.transform(
   Seq("Clara often travels between New York and Paris.").toDS.toDF("text"))

 results
   .selectExpr("document", "explode(zero_shot_ner) AS entity")
   .select(
     col("entity.result"),
     col("entity.metadata.word"),
     col("entity.metadata.sentence"),
     col("entity.begin"),
     col("entity.end"),
     col("entity.metadata.confidence"),
     col("entity.metadata.question"))
   .show(truncate=false)

+------+-----+--------+-----+---+----------+------------------+
|result|word |sentence|begin|end|confidence|question          |
+------+-----+--------+-----+---+----------+------------------+
|B-CITY|Paris|0       |41   |45 |0.78655756|Which is the city?|
|B-CITY|New  |0       |28   |30 |0.29346612|Which city?       |
|I-CITY|York |0       |32   |35 |0.29346612|Which city?       |
+------+-----+--------+-----+---+----------+------------------+

{%- endcapture -%}

{%- capture api_link -%}
[ZeroShotNerModel](/api/com/johnsnowlabs/nlp/annotators/ner/dl/ZeroShotNerModel)
{%- endcapture -%}

{%- capture python_api_link -%}
[ZeroShotNerModel](/api/python/reference/autosummary/sparknlp/annotator/ner/zero_shot_ner_model/index.html#sparknlp.annotator.ner.zero_shot_ner_model.ZeroShotNerModel)
{%- endcapture -%}

{%- capture source_link -%}
[ZeroShotNerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/dl/ZeroShotNerModel.scala)
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