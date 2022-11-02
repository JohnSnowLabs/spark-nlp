{%- capture title -%}
ContextSpellChecker
{%- endcapture -%}

{%- capture model_description -%}
Implements a deep-learning based Noisy Channel Model Spell Algorithm.
Correction candidates are extracted combining context information and word information.

Spell Checking is a sequence to sequence mapping problem. Given an input sequence, potentially containing a
certain number of errors, `ContextSpellChecker` will rank correction sequences according to three things:
 1. Different correction candidates for each word — **word level**.
 1. The surrounding text of each word, i.e. it’s context — **sentence level**.
 1. The relative cost of different correction candidates according to the edit operations at the character level it requires — **subword level**.

For an in-depth explanation of the module see the article [Applying Context Aware Spell Checking in Spark NLP](https://medium.com/spark-nlp/applying-context-aware-spell-checking-in-spark-nlp-3c29c46963bc).

This is the instantiated model of the ContextSpellCheckerApproach.
For training your own model, please see the documentation of that class.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val spellChecker = ContextSpellCheckerModel.pretrained()
  .setInputCols("token")
  .setOutputCol("checked")
```
The default model is `"spellcheck_dl"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Spell+Check).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SPELL_CHECKER_EN.ipynb)
and the [ContextSpellCheckerTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerTestSpec.scala).
{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("doc")

tokenizer = Tokenizer() \
    .setInputCols(["doc"]) \
    .setOutputCol("token")

spellChecker = ContextSpellCheckerModel \
    .pretrained() \
    .setTradeoff(12.0) \
    .setInputCols("token") \
    .setOutputCol("checked")

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    spellChecker
])

data = spark.createDataFrame([["It was a cold , dreary day and the country was white with smow ."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("checked.result").show(truncate=False)
+--------------------------------------------------------------------------------+
|result                                                                          |
+--------------------------------------------------------------------------------+
|[It, was, a, cold, ,, dreary, day, and, the, country, was, white, with, snow, .]|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerModel
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("doc")

val tokenizer = new Tokenizer()
  .setInputCols(Array("doc"))
  .setOutputCol("token")

val spellChecker = ContextSpellCheckerModel
  .pretrained()
  .setTradeOff(12.0f)
  .setInputCols("token")
  .setOutputCol("checked")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  spellChecker
))

val data = Seq("It was a cold , dreary day and the country was white with smow .").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("checked.result").show(false)
+--------------------------------------------------------------------------------+
|result                                                                          |
+--------------------------------------------------------------------------------+
|[It, was, a, cold, ,, dreary, day, and, the, country, was, white, with, snow, .]|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[ContextSpellCheckerModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[ContextSpellCheckerModel](/api/python/reference/autosummary/python/sparknlp/annotator/spell_check/context_spell_checker/index.html#sparknlp.annotator.spell_check.context_spell_checker.ContextSpellCheckerModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[ContextSpellCheckerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a deep-learning based Noisy Channel Model Spell Algorithm.
Correction candidates are extracted combining context information and word information.

For instantiated/pretrained models, see ContextSpellCheckerModel.

Spell Checking is a sequence to sequence mapping problem. Given an input sequence, potentially containing a
certain number of errors, `ContextSpellChecker` will rank correction sequences according to three things:
 1. Different correction candidates for each word — **word level**.
 1. The surrounding text of each word, i.e. it’s context — **sentence level**.
 1. The relative cost of different correction candidates according to the edit operations at the character level it requires — **subword level**.

For an in-depth explanation of the module see the article [Applying Context Aware Spell Checking in Spark NLP](https://medium.com/spark-nlp/applying-context-aware-spell-checking-in-spark-nlp-3c29c46963bc).

For extended examples of usage, see the article [Training a Contextual Spell Checker for Italian Language](https://towardsdatascience.com/training-a-contextual-spell-checker-for-italian-language-66dda528e4bf),
the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/blogposts/5.TrainingContextSpellChecker.ipynb)
and the [ContextSpellCheckerTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerTestSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_python_example -%}
# For this example, we use the first Sherlock Holmes book as the training dataset.

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")


tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

spellChecker = ContextSpellCheckerApproach() \
    .setInputCols("token") \
    .setOutputCol("corrected") \
    .setWordMaxDistance(3) \
    .setBatchSize(24) \
    .setEpochs(8) \
    .setLanguageModelClasses(1650)  # dependant on vocabulary size
    # .addVocabClass("_NAME_", names) # Extra classes for correction could be added like this

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    spellChecker
])

path = "sherlockholmes.txt"
dataset = spark.read.text(path) \
    .toDF("text")
pipelineModel = pipeline.fit(dataset)

{%- endcapture -%}

{%- capture approach_scala_example -%}
// For this example, we use the first Sherlock Holmes book as the training dataset.

import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerApproach

import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")


val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val spellChecker = new ContextSpellCheckerApproach()
  .setInputCols("token")
  .setOutputCol("corrected")
  .setWordMaxDistance(3)
  .setBatchSize(24)
  .setEpochs(8)
  .setLanguageModelClasses(1650)  // dependant on vocabulary size
  // .addVocabClass("_NAME_", names) // Extra classes for correction could be added like this

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  spellChecker
))

val path = "src/test/resources/spell/sherlockholmes.txt"
val dataset = spark.sparkContext.textFile(path)
  .toDF("text")
val pipelineModel = pipeline.fit(dataset)

{%- endcapture -%}

{%- capture approach_api_link -%}
[ContextSpellCheckerApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[ContextSpellCheckerApproach](/api/python/reference/autosummary/python/sparknlp/annotator/spell_check/context_spell_checker/index.html#sparknlp.annotator.spell_check.context_spell_checker.ContextSpellCheckerApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[ContextSpellCheckerApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerApproach.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_api_link=model_python_api_link
model_api_link=model_api_link
model_source_link=model_source_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_python_api_link=approach_python_api_link
approach_api_link=approach_api_link
approach_source_link=approach_source_link
%}
