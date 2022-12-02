{%- capture title -%}
NerOverwriter
{%- endcapture -%}

{%- capture description -%}
Overwrites entities of specified strings.

The input for this Annotator have to be entities that are already extracted, Annotator type `NAMED_ENTITY`.
The strings specified with `setStopWords` will have new entities assigned to, specified with `setNewResult`.
{%- endcapture -%}

{%- capture input_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# First extract the prerequisite Entities
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("bert")

nerTagger = NerDLModel.pretrained() \
    .setInputCols(["sentence", "token", "bert"]) \
    .setOutputCol("ner")

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    embeddings,
    nerTagger
])

data = spark.createDataFrame([["Spark NLP Crosses Five Million Downloads, John Snow Labs Announces."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(ner)").show(truncate=False)
# +------------------------------------------------------+
# |col                                                   |
# +------------------------------------------------------+
# |[named_entity, 0, 4, B-ORG, [word -> Spark], []]      |
# |[named_entity, 6, 8, I-ORG, [word -> NLP], []]        |
# |[named_entity, 10, 16, O, [word -> Crosses], []]      |
# |[named_entity, 18, 21, O, [word -> Five], []]         |
# |[named_entity, 23, 29, O, [word -> Million], []]      |
# |[named_entity, 31, 39, O, [word -> Downloads], []]    |
# |[named_entity, 40, 40, O, [word -> ,], []]            |
# |[named_entity, 42, 45, B-ORG, [word -> John], []]     |
# |[named_entity, 47, 50, I-ORG, [word -> Snow], []]     |
# |[named_entity, 52, 55, I-ORG, [word -> Labs], []]     |
# |[named_entity, 57, 65, I-ORG, [word -> Announces], []]|
# |[named_entity, 66, 66, O, [word -> .], []]            |
# +------------------------------------------------------+

# The recognized entities can then be overwritten
nerOverwriter = NerOverwriter() \
    .setInputCols(["ner"]) \
    .setOutputCol("ner_overwritten") \
    .setStopWords(["Million"]) \
    .setNewResult("B-CARDINAL")

nerOverwriter.transform(result).selectExpr("explode(ner_overwritten)").show(truncate=False)
+---------------------------------------------------------+
|col                                                      |
+---------------------------------------------------------+
|[named_entity, 0, 4, B-ORG, [word -> Spark], []]         |
|[named_entity, 6, 8, I-ORG, [word -> NLP], []]           |
|[named_entity, 10, 16, O, [word -> Crosses], []]         |
|[named_entity, 18, 21, O, [word -> Five], []]            |
|[named_entity, 23, 29, B-CARDINAL, [word -> Million], []]|
|[named_entity, 31, 39, O, [word -> Downloads], []]       |
|[named_entity, 40, 40, O, [word -> ,], []]               |
|[named_entity, 42, 45, B-ORG, [word -> John], []]        |
|[named_entity, 47, 50, I-ORG, [word -> Snow], []]        |
|[named_entity, 52, 55, I-ORG, [word -> Labs], []]        |
|[named_entity, 57, 65, I-ORG, [word -> Announces], []]   |
|[named_entity, 66, 66, O, [word -> .], []]               |
+---------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import com.johnsnowlabs.nlp.annotators.ner.NerOverwriter
import org.apache.spark.ml.Pipeline

// First extract the prerequisite Entities
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("bert")

val nerTagger = NerDLModel.pretrained()
  .setInputCols("sentence", "token", "bert")
  .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerTagger
))

val data = Seq("Spark NLP Crosses Five Million Downloads, John Snow Labs Announces.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(ner)").show(false)
/
+------------------------------------------------------+
|col                                                   |
+------------------------------------------------------+
|[named_entity, 0, 4, B-ORG, [word -> Spark], []]      |
|[named_entity, 6, 8, I-ORG, [word -> NLP], []]        |
|[named_entity, 10, 16, O, [word -> Crosses], []]      |
|[named_entity, 18, 21, O, [word -> Five], []]         |
|[named_entity, 23, 29, O, [word -> Million], []]      |
|[named_entity, 31, 39, O, [word -> Downloads], []]    |
|[named_entity, 40, 40, O, [word -> ,], []]            |
|[named_entity, 42, 45, B-ORG, [word -> John], []]     |
|[named_entity, 47, 50, I-ORG, [word -> Snow], []]     |
|[named_entity, 52, 55, I-ORG, [word -> Labs], []]     |
|[named_entity, 57, 65, I-ORG, [word -> Announces], []]|
|[named_entity, 66, 66, O, [word -> .], []]            |
+------------------------------------------------------+
/
// The recognized entities can then be overwritten
val nerOverwriter = new NerOverwriter()
  .setInputCols("ner")
  .setOutputCol("ner_overwritten")
  .setStopWords(Array("Million"))
  .setNewResult("B-CARDINAL")

nerOverwriter.transform(result).selectExpr("explode(ner_overwritten)").show(false)
+---------------------------------------------------------+
|col                                                      |
+---------------------------------------------------------+
|[named_entity, 0, 4, B-ORG, [word -> Spark], []]         |
|[named_entity, 6, 8, I-ORG, [word -> NLP], []]           |
|[named_entity, 10, 16, O, [word -> Crosses], []]         |
|[named_entity, 18, 21, O, [word -> Five], []]            |
|[named_entity, 23, 29, B-CARDINAL, [word -> Million], []]|
|[named_entity, 31, 39, O, [word -> Downloads], []]       |
|[named_entity, 40, 40, O, [word -> ,], []]               |
|[named_entity, 42, 45, B-ORG, [word -> John], []]        |
|[named_entity, 47, 50, I-ORG, [word -> Snow], []]        |
|[named_entity, 52, 55, I-ORG, [word -> Labs], []]        |
|[named_entity, 57, 65, I-ORG, [word -> Announces], []]   |
|[named_entity, 66, 66, O, [word -> .], []]               |
+---------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[NerOverwriter](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ner/NerOverwriter)
{%- endcapture -%}

{%- capture python_api_link -%}
[NerOverwriter](/api/python/reference/autosummary/python/sparknlp/annotator/ner/ner_overwriter/index.html#sparknlp.annotator.ner.ner_overwriter.NerOverwriter)
{%- endcapture -%}

{%- capture source_link -%}
[NerOverwriter](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/NerOverwriter.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
%}