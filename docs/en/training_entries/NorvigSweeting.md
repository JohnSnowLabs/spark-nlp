{%- capture title -%}
NorvigSweeting Spellchecker
{%- endcapture -%}

{%- capture description -%}
Trains annotator, that retrieves tokens and makes corrections automatically if not found in an English dictionary.

The Symmetric Delete spelling correction algorithm reduces the complexity of edit candidate generation and
dictionary lookup for a given Damerau-Levenshtein distance. It is six orders of magnitude faster
(than the standard approach with deletes + transposes + replaces + inserts) and language independent.
A dictionary of correct spellings must be provided with `setDictionary` as a text file, where each word is parsed by a regex pattern.

For Example a file `"words.txt"`:
```
...
gummy
gummic
gummier
gummiest
gummiferous
...
```
can be parsed with the regular expression `\S+`, which is the default for `setDictionary`.

This dictionary is then set to be the basis of the spell checker.

Inspired by Norvig model and [SymSpell](https://github.com/wolfgarbe/SymSpell).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

spellChecker = NorvigSweetingApproach() \
    .setInputCols(["token"]) \
    .setOutputCol("spell") \
    .setDictionary("src/test/resources/spell/words.txt")

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    spellChecker
])

pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val spellChecker = new NorvigSweetingApproach()
  .setInputCols("token")
  .setOutputCol("spell")
  .setDictionary("src/test/resources/spell/words.txt")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  spellChecker
))

val pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture api_link -%}
[NorvigSweetingApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[NorvigSweetingApproach](/api/python/reference/autosummary/python/sparknlp/annotator/spell_check/norvig_sweeting/index.html#sparknlp.annotator.spell_check.norvig_sweeting.NorvigSweetingApproach)
{%- endcapture -%}

{%- capture source_link -%}
[NorvigSweetingApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingApproach.scala)
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
