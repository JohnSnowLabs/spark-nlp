{%- capture title -%}
CoHereTransformer
{%- endcapture -%}

{%- capture description -%}
Text Generation using Cohere Command-R.

C4AI Command-R is a research release of a 35 billion parameter highly performant generative model.
Command-R is a large language model with open weights optimized for a variety of use cases including reasoning,
summarization, and question answering. Command-R has the capability for multilingual generation evaluated
in 10 languages and highly performant RAG capabilities.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val CoHere = CoHereTransformer.pretrained()
     .setInputCols("document")
     .setOutputCol("generation")
```
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler()   
          .setInputCol("text")   
          .setOutputCol("documents")
CoHere = CoHereTransformer.pretrained("c4ai_command_r_v01_int4","en")   
          .setInputCols(["documents"])   
          .setMaxOutputLength(60)   
          .setOutputCol("generation")
pipeline = Pipeline().setStages([documentAssembler, CoHere])
data = spark.createDataFrame([
              (
                        1,
                        "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
                        )
          ]).toDF("id", "text")
result = pipeline.fit(data).transform(data)
result.select("generation.result").show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.CoHereTransformer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
         .setInputCol("text")
         .setOutputCol("documents")

val CoHere = CoHereTransformer.pretrained("c4ai_command_r_v01_int4")
         .setInputCols(Array("documents"))
         .setMinOutputLength(15)
         .setMaxOutputLength(60)
         .setDoSample(false)
         .setTopK(40)
         .setNoRepeatNgramSize(3)
         .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, CoHere))

val data = Seq(
     (
              1,
              """
              <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
              """.stripMargin)
).toDF("id", "text")

val result = pipeline.fit(data).transform(data)

result.select("generation.result").show(truncate = false)
{%- endcapture -%}

{%- capture api_link -%}
[CoHereTransformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/CoHereTransformer)
{%- endcapture -%}

{%- capture python_api_link -%}
[CoHereTransformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/cohere/index.html#sparknlp.annotator.seq2seq.cohere.CoHereTransformer)
{%- endcapture -%}

{%- capture source_link -%}
[CoHereTransformer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/CoHereTransformer.scala)
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