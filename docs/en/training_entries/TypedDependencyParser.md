{%- capture title -%}
TypedDependencyParser
{%- endcapture -%}

{%- capture description -%}
Labeled parser that finds a grammatical relation between two words in a sentence.
Its input is either a CoNLL2009 or ConllU dataset.

For instantiated/pretrained models, see TypedDependencyParserModel.

Dependency parsers provide information about word relationship. For example, dependency parsing can tell you what
the subjects and objects of a verb are, as well as which words are modifying (describing) the subject. This can help
you find precise answers to specific questions.

The parser requires the dependant tokens beforehand with e.g. [DependencyParser](/docs/en/annotators#dependencyparser).
The required training data can be set in two different ways (only one can be chosen for a particular model):
  - Dataset in the [CoNLL 2009 format](https://ufal.mff.cuni.cz/conll2009-st/trial-data.html) set with `setConll2009`. Data format:
    ```
    1	The	the	the	DT	DT	_	_	4	4	NMOD	NMOD	_	_	_	_
    2	most	most	most	RBS	RBS	_	_	3	3	AMOD	AMOD	_	_	_	_
    3	troublesome	troublesome	troublesome	JJ	JJ	_	_	4	4	NMOD	NMOD	_	_	_	_
    ```
  - Dataset in the [CoNLL-U format](https://universaldependencies.org/format.html) set with `setConllU`
    Data format:
    ```
    -DOCSTART- -X- -X- O

    EU NNP B-NP B-ORG
    rejects VBZ B-VP O
    German JJ B-NP B-MISC
    ```

Apart from that, no additional training data is needed.

See [TypedDependencyParserApproachTestSpec](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserApproachTestSpec.scala) for further reference on this API.
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN, POS, DEPENDENCY
{%- endcapture -%}

{%- capture output_anno -%}
LABELED_DEPENDENCY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

posTagger = PerceptronModel.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")

dependencyParser = DependencyParserModel.pretrained() \
    .setInputCols(["sentence", "pos", "token"]) \
    .setOutputCol("dependency")

typedDependencyParser = TypedDependencyParserApproach() \
    .setInputCols(["dependency", "pos", "token"]) \
    .setOutputCol("dependency_type") \
    .setConllU("src/test/resources/parser/labeled/train_small.conllu.txt") \
    .setNumberOfIterations(1)

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    posTagger,
    dependencyParser,
    typedDependencyParser
])

# Additional training data is not needed, the dependency parser relies on CoNLL-U only.
emptyDataSet = spark.createDataFrame([[""]]).toDF("text")
pipelineModel = pipeline.fit(emptyDataSet)

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserApproach
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val posTagger = PerceptronModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("pos")

val dependencyParser = DependencyParserModel.pretrained()
  .setInputCols("sentence", "pos", "token")
  .setOutputCol("dependency")

val typedDependencyParser = new TypedDependencyParserApproach()
  .setInputCols("dependency", "pos", "token")
  .setOutputCol("dependency_type")
  .setConllU("src/test/resources/parser/labeled/train_small.conllu.txt")
  .setNumberOfIterations(1)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  posTagger,
  dependencyParser,
  typedDependencyParser
))

// Additional training data is not needed, the dependency parser relies on CoNLL-U only.
val emptyDataSet = Seq.empty[String].toDF("text")
val pipelineModel = pipeline.fit(emptyDataSet)

{%- endcapture -%}

{%- capture api_link -%}
[TypedDependencyParserApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[TypedDependencyParserApproach](/api/python/reference/autosummary/python/sparknlp/annotator/dependency/typed_dependency_parser/index.html#sparknlp.annotator.dependency.typed_dependency_parser.TypedDependencyParserApproach)
{%- endcapture -%}

{%- capture source_link -%}
[TypedDependencyParserApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserApproach.scala)
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
