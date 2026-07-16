{%- capture title -%}
RuleBasedMatcher
{%- endcapture -%}

{%- capture model_description -%}
Instantiated model of the RuleBasedMatcher.
For usage and examples see the documentation of the main class.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_api_link -%}
[RuleBasedMatcherModel](/api/com/johnsnowlabs/nlp/annotators/matcher/RuleBasedMatcherModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[RuleBasedMatcherModel](/api/python/reference/autosummary/sparknlp/annotator/matcher/rule_based_matcher/index.html#sparknlp.annotator.matcher.rule_based_matcher.RuleBasedMatcherModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[RuleBasedMatcherModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/matcher/RuleBasedMatcherModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
`RuleBasedMatcher` matches token spans with declarative rules over Spark NLP
annotation columns. It is useful when a rule needs more than token text, for
example a sequence that combines token text, POS tags, lemmas, NER tags,
dependency metadata, or custom annotation metadata.

The minimum inputs are one `DOCUMENT` column and one base `TOKEN` column. Other
annotation columns are optional and are used only when rules reference their
attributes. Matches are emitted as `CHUNK` annotations.

For a complete advanced tutorial with postal address extraction, contact
information, job requirements, business phrases, domain terminology, external
rule files, persistence, and rule diagnostics, see the
[RuleBasedMatcher notebook](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/rule-based-matcher/RuleBasedMatcher.ipynb).

Use `setAttributeColumns` to map rule attributes to annotation columns:

```json
{
  "TEXT": "token",
  "LOWER": "token",
  "POS": "pos",
  "LEMMA": "lemma",
  "NER": "ner",
  "NER_TYPE": "ner"
}
```

Built-in attributes:

* `TEXT`, `TOKEN`, `LOWER`, `LENGTH`: values from the base `TOKEN` column.
* `LEMMA`: values from a lemma or normalized-token column. Spark NLP lemmatizers
  output `TOKEN`, so map `LEMMA` explicitly when both token and lemma columns are
  present.
* `POS`: values from a `POS` column.
* `NER` and `NER_TAG`: raw `NAMED_ENTITY` tags such as `B-GPE`, `I-GPE`,
  `U-GPE`, or `O`. `NER` remains raw for backward-compatible, unsurprising
  behavior.
* `NER_TYPE` and `ENT_TYPE`: normalized named entity types with common BIO,
  IOB, BIOES, and BILOU prefixes removed, for example `B-GPE` -> `GPE`.
  This does not validate complete entity boundaries; it matches token-level
  normalized types.
* `DEP`, `HEAD`, `HEAD_BEGIN`, `HEAD_END`: dependency values and metadata from a
  `DEPENDENCY` column.
* `DEP_LABEL`, `RELATION`: values from a `LABELED_DEPENDENCY` column.
* `META.key`: metadata value `key` from the mapped annotation.

Rules are JSON or JSONL. A rule has an `id`, optional `label`, optional integer
`priority`, and one or more token patterns:

```json
[
  {
    "id": "address_like_location",
    "label": "ADDRESS",
    "priority": 10,
    "patterns": [[
      {"POS": {"IN": ["NUM", "CD"]}},
      {"TEXT": {"REGEX": "^[0-9]+(?:st|nd|rd|th)?$"}, "OP": "?"},
      {"POS": {"IN": ["NOUN", "PROPN", "NN", "NNP"]}},
      {"NER_TYPE": {"IN": ["GPE", "LOC"]}, "OP": "+"}
    ]]
  }
]
```

Supported predicates are exact equality, `IN`, `NOT_IN`, `REGEX`,
`NOT_REGEX`, and `EXISTS`. Multiple predicates on one token are combined with
logical AND. Missing attributes behave explicitly: equality, `IN`, regex, and
negated predicates do not match when the attribute is absent. Use
`{"ATTRIBUTE": {"EXISTS": false}}` to intentionally match missing values.

Supported token quantifiers are:

* omitted or empty: exactly one token.
* `?`: zero or one token.
* `*`: zero or more tokens.
* `+`: one or more tokens.
* `{n}`: exactly `n` tokens.
* `{n,m}` and `{n,}`: bounded or open upper-bound repetitions.

The canonical wildcard token is `{}`. It matches any one token. Add `OP` to
repeat wildcards, for example `{"OP": "*"}` for any number of tokens. Adjacent
unbounded wildcard repetitions are rejected because they can create pathological
matching behavior.

Alignment and boundaries:

* The matcher builds a token grid from the configured base `TOKEN` column.
* Default `alignmentMode` is `STRICT`, which aligns attributes by document,
  sentence, begin, and end offsets.
* `POSITIONAL` alignment may be used for legacy/custom pipelines where offsets
  are unreliable but sentence-local token counts match.
* Matches never cross sentence or document boundaries.
* If multiple `TOKEN` input columns are present, map `TEXT`, `TOKEN`, or
  `LOWER` to the base token column explicitly.

Overlap resolution is controlled by `setOverlapStrategy`:

* `ALL`: return every candidate match from every starting position. Within each
  starting position, the last pattern element greedily matches its maximum span.
  To enumerate all possible span lengths from the same start, use multiple
  explicit patterns with exact quantifiers such as `{1}`, `{2}`, and `{3}`.
* `FIRST`: keep the earliest non-overlapping candidates in rule order.
* `LONGEST`: prefer longer spans.
* `PRIORITY_LONGEST`: prefer higher rule priority, then longer spans.

Output annotations have type `CHUNK`. Metadata includes `entity`, `label`,
`rule`, `priority`, `pattern`, `sentence`, `chunk`, `documentKey`,
`tokenBegin`, `tokenEnd`, `sentenceTokenBegin`, `sentenceTokenEnd`,
`documentTokenBegin`, and `documentTokenEnd`. `tokenBegin` and `tokenEnd` are
kept for compatibility and are sentence-local; prefer the explicit
`sentenceToken*` and `documentToken*` fields in new code.

`RuleBasedMatcher` supports Spark DataFrame pipelines, saved/loaded models,
saved/loaded `PipelineModel`s, and `LightPipeline`. DataFrame and LightPipeline
execution preserve input-column provenance internally. Direct calls to
`RuleBasedMatcherModel.annotate` with flattened annotations are supported only
when source columns are unambiguous; if multiple input columns share the same
annotation type, provide `source_column` metadata or run through a pipeline.
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture approach_python_example -%}
from pyspark.ml import Pipeline
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import (
    SentenceDetector,
    Tokenizer,
    PerceptronModel,
    WordEmbeddingsModel,
    NerCrfModel,
    RuleBasedMatcher,
)

data = spark.createDataFrame([["443 8th Street New York"]]).toDF("text")

document = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

token = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

pos = PerceptronModel.pretrained("pos_anc", "en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")

embeddings = WordEmbeddingsModel.pretrained("glove_100d", "en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(False)

ner = NerCrfModel.pretrained("ner_crf", "en") \
    .setInputCols(["sentence", "token", "pos", "embeddings"]) \
    .setOutputCol("ner")

rules = [
    {
        "id": "address_like_location",
        "label": "ADDRESS",
        "priority": 10,
        "patterns": [[
            {"POS": {"IN": ["NUM", "CD"]}},
            {"POS": {"IN": ["NUM", "CD"]}},
            {"POS": {"IN": ["NOUN", "PROPN", "NN", "NNP"]}},
            {"NER_TYPE": {"IN": ["GPE", "LOC"]}, "OP": "+"}
        ]]
    }
]

matcher = RuleBasedMatcher() \
    .setInputCols(["document", "sentence", "token", "pos", "ner"]) \
    .setOutputCol("rule_matches") \
    .setRules(rules) \
    .setAttributeColumns({
        "TEXT": "token",
        "LOWER": "token",
        "POS": "pos",
        "NER": "ner",
        "NER_TYPE": "ner"
    }) \
    .setOverlapStrategy("ALL")

pipeline = Pipeline(stages=[document, sentence, token, pos, embeddings, ner, matcher])
result = pipeline.fit(data).transform(data)
result.selectExpr("explode(rule_matches) as match").select("match.result", "match.metadata").show(truncate=False)
{%- endcapture -%}

{%- capture approach_scala_example -%}
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val data = Seq("443 8th Street New York").toDF("text")

val document = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val token = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val pos = PerceptronModel.pretrained("pos_anc", "en")
  .setInputCols("sentence", "token")
  .setOutputCol("pos")

val embeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")
  .setCaseSensitive(false)

val ner = NerCrfModel.pretrained("ner_crf", "en")
  .setInputCols("sentence", "token", "pos", "embeddings")
  .setOutputCol("ner")

val rules =
  """
    |[
    |  {
    |    "id": "address_like_location",
    |    "label": "ADDRESS",
    |    "priority": 10,
    |    "patterns": [[
    |      {"POS": {"IN": ["NUM", "CD"]}},
    |      {"POS": {"IN": ["NUM", "CD"]}},
    |      {"POS": {"IN": ["NOUN", "PROPN", "NN", "NNP"]}},
    |      {"NER_TYPE": {"IN": ["GPE", "LOC"]}, "OP": "+"}
    |    ]]
    |  }
    |]
    |""".stripMargin

val matcher = new RuleBasedMatcher()
  .setInputCols("document", "sentence", "token", "pos", "ner")
  .setOutputCol("rule_matches")
  .setRules(rules)
  .setAttributeColumns(Map(
    "TEXT" -> "token",
    "LOWER" -> "token",
    "POS" -> "pos",
    "NER" -> "ner",
    "NER_TYPE" -> "ner"))
  .setOverlapStrategy("ALL")

val pipeline = new Pipeline().setStages(Array(document, sentence, token, pos, embeddings, ner, matcher))
val result = pipeline.fit(data).transform(data)
result.selectExpr("explode(rule_matches) as match").select("match.result", "match.metadata").show(false)
{%- endcapture -%}

{%- capture approach_api_link -%}
[RuleBasedMatcher](/api/com/johnsnowlabs/nlp/annotators/matcher/RuleBasedMatcher)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[RuleBasedMatcher](/api/python/reference/autosummary/sparknlp/annotator/matcher/rule_based_matcher/index.html#sparknlp.annotator.matcher.rule_based_matcher.RuleBasedMatcher)
{%- endcapture -%}

{%- capture approach_source_link -%}
[RuleBasedMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/matcher/RuleBasedMatcher.scala)
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
