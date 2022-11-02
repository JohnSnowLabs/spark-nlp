{%- capture title -%}
ContextSpellCheckerApproach
{%- endcapture -%}

{%- capture description -%}
Trains a deep-learning based Noisy Channel Model Spell Algorithm.
Correction candidates are extracted combining context information and word information.

Spell Checking is a sequence to sequence mapping problem. Given an input sequence, potentially containing a
certain number of errors, `ContextSpellChecker` will rank correction sequences according to three things:
 1. Different correction candidates for each word — **word level**.
 2. The surrounding text of each word, i.e. it’s context — **sentence level**.
 3. The relative cost of different correction candidates according to the edit operations at the character level it requires — **subword level**.

For an in-depth explanation of the module see the article [Applying Context Aware Spell Checking in Spark NLP](https://medium.com/spark-nlp/applying-context-aware-spell-checking-in-spark-nlp-3c29c46963bc).

For extended examples of usage, see the article [Training a Contextual Spell Checker for Italian Language](https://towardsdatascience.com/training-a-contextual-spell-checker-for-italian-language-66dda528e4bf),
the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/blogposts/5.TrainingContextSpellChecker.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture python_example -%}
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

{%- capture scala_example -%}
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

{%- capture api_link -%}
[ContextSpellCheckerApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[ContextSpellCheckerApproach](/api/python/reference/autosummary/python/sparknlp/annotator/spell_check/context_spell_checker/index.html#sparknlp.annotator.spell_check.context_spell_checker.ContextSpellCheckerApproach)
{%- endcapture -%}

{%- capture source_link -%}
[ContextSpellCheckerApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerApproach.scala)
{%- endcapture -%}

{% include templates/training_anno_template.md
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
