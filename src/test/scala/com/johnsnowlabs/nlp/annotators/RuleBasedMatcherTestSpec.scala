/*
 * Copyright 2017-2026 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.rulematcher.MatchCandidate
import com.johnsnowlabs.nlp.annotators.matcher.{RuleBasedMatcher, RuleBasedMatcherModel}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.nlp.util.io.{ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.MetadataBuilder
import org.scalatest.flatspec.AnyFlatSpec

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import scala.reflect.io.Directory

class RuleBasedMatcherTestSpec extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  "RuleBasedMatcher" should "match combined POS and normalized NER type patterns" taggedAs FastTest in {
    val dataset = addressDataset()
    val matcher = new RuleBasedMatcher()
      .setInputCols("sentence", "token", "pos", "ner")
      .setOutputCol("matches")
      .setRules(addressRules(attribute = "NER_TYPE"))
      .setAttributeColumns(Map("TEXT" -> "token", "POS" -> "pos", "NER_TYPE" -> "ner"))

    val matches = collectMatches(matcher.fit(dataset).transform(dataset))

    assert(matches.map(_.result) == Seq("443 8th Street New York"))
    assert(matches.head.metadata("rule") == "address_like_location")
    assert(matches.head.metadata("entity") == "ADDRESS")
    assert(matches.head.metadata("sentenceTokenBegin") == "0")
    assert(matches.head.metadata("sentenceTokenEnd") == "4")
    assert(matches.head.metadata("documentTokenBegin") == "0")
    assert(matches.head.metadata("documentTokenEnd") == "4")
    assert(matches.head.metadata("tokenBegin") == "0")
    assert(matches.head.metadata("tokenEnd") == "4")
  }

  it should "distinguish raw NER tags from normalized NER types" taggedAs FastTest in {
    val dataset = addressDataset()
    val rawMatcher = new RuleBasedMatcher()
      .setInputCols("sentence", "token", "ner")
      .setOutputCol("matches")
      .setRules(singleTokenRule("raw_gpe", "PLACE", "NER_TAG", "B-GPE"))
      .setAttributeColumns(Map("TEXT" -> "token", "NER_TAG" -> "ner"))

    val typeMatcher = new RuleBasedMatcher()
      .setInputCols("sentence", "token", "ner")
      .setOutputCol("matches")
      .setRules(singleTokenRule("type_gpe", "PLACE", "NER_TYPE", "GPE"))
      .setAttributeColumns(Map("TEXT" -> "token", "NER_TYPE" -> "ner"))

    assert(collectMatches(rawMatcher.fit(dataset).transform(dataset)).map(_.result) == Seq("New"))
    assert(
      collectMatches(typeMatcher.fit(dataset).transform(dataset))
        .map(_.result) == Seq("New", "York"))
  }

  it should "match exact token text, lemma token columns, and combined attributes" taggedAs FastTest in {
    val dataset = lemmaDataset()
    val rules =
      """
        |[
        |  {"id":"exact_text","patterns":[[{"TEXT":"Dogs"}]]},
        |  {"id":"lemma_pos","patterns":[[{"LEMMA":"dog","POS":"NOUN"},{"LEMMA":"bark","POS":"VERB"}]]}
        |]
        |""".stripMargin

    val matcher = new RuleBasedMatcher()
      .setInputCols("sentence", "token", "lemma", "pos")
      .setOutputCol("matches")
      .setRules(rules)
      .setAttributeColumns(Map("TEXT" -> "token", "LEMMA" -> "lemma", "POS" -> "pos"))

    val matches = collectMatches(matcher.fit(dataset).transform(dataset))
    assert(matches.map(_.metadata("rule")) == Seq("exact_text", "lemma_pos"))
    assert(matches.map(_.result) == Seq("Dogs", "Dogs bark"))
  }

  it should "infer the base token column automatically without an explicit attributeColumns mapping" taggedAs FastTest in {
    val dataset = simpleDataset("dogs bark")
    val matcher = new RuleBasedMatcher()
      .setInputCols("sentence", "token")
      .setOutputCol("matches")
      .setRules("""[{"id":"dogs_rule","patterns":[[{"TEXT":"dogs"}]]}]""")

    val matches = collectMatches(matcher.fit(dataset).transform(dataset))
    assert(matches.map(_.result) == Seq("dogs"))
  }

  it should "support wildcard, optional, star, plus, bounded, regex, and multi-predicate rules" taggedAs FastTest in {
    val dataset = simpleDataset("red very big car", Seq("ADJ", "ADV", "ADJ", "NOUN"))
    val rules =
      """
        |[
        |  {"id":"wildcard","label":"THING","patterns":[[
        |    {"LOWER":"red"},
        |    {"OP":"*"},
        |    {"TEXT":{"REGEX":"^car$"}}
        |  ]]},
        |  {"id":"bounded","label":"THING","patterns":[[
        |    {"POS":"ADJ"},
        |    {"POS":{"NOT_IN":["NOUN"]},"OP":"{1,2}"},
        |    {"POS":"NOUN","TEXT":"car"}
        |  ]]},
        |  {"id":"optional","label":"THING","patterns":[[
        |    {"LOWER":"red"},
        |    {"LOWER":"very","OP":"?"},
        |    {"POS":"ADJ","OP":"+"},
        |    {"LOWER":{"NOT_REGEX":"^truck$"}}
        |  ]]}
        |]
        |""".stripMargin

    val matcher = new RuleBasedMatcher()
      .setInputCols("sentence", "token", "pos")
      .setOutputCol("matches")
      .setRules(rules)
      .setAttributeColumns(Map("TEXT" -> "token", "LOWER" -> "token", "POS" -> "pos"))

    val matches = collectMatches(matcher.fit(dataset).transform(dataset))
    assert(matches.map(_.metadata("rule")).toSet == Set("wildcard", "bounded", "optional"))
    assert(matches.forall(_.result == "red very big car"))
  }

  it should "treat missing attributes safely for negation and EXISTS predicates" taggedAs FastTest in {
    val dataset = datasetWithEmptyNer("dogs bark", Seq("NOUN", "VERB"))
    val rules =
      """
        |[
        |  {"id":"not_missing","patterns":[[{"NER":{"NOT_IN":["O"]}}]]},
        |  {"id":"missing_exists","patterns":[[{"NER":{"EXISTS":false}}]]},
        |  {"id":"pos_exists","patterns":[[{"POS":{"EXISTS":true}}, {"POS":{"NOT_IN":["NOUN"]}}]]}
        |]
        |""".stripMargin

    val matcher = new RuleBasedMatcher()
      .setInputCols("sentence", "token", "pos", "ner")
      .setOutputCol("matches")
      .setRules(rules)
      .setAttributeColumns(Map("TEXT" -> "token", "POS" -> "pos", "NER" -> "ner"))

    val matches = collectMatches(matcher.fit(dataset).transform(dataset))
    assert(!matches.exists(_.metadata("rule") == "not_missing"))
    assert(matches.count(_.metadata("rule") == "missing_exists") == 2)
    assert(matches.exists(m => m.metadata("rule") == "pos_exists" && m.result == "dogs bark"))
  }

  it should "not match NOT_REGEX predicates when the attribute is absent" taggedAs FastTest in {
    val dataset = datasetWithEmptyNer("dogs bark", Seq("NOUN", "VERB"))
    val matcher = new RuleBasedMatcher()
      .setInputCols("sentence", "token", "pos", "ner")
      .setOutputCol("matches")
      .setRules("""[{"id":"missing_not_regex","patterns":[[{"NER":{"NOT_REGEX":"^O$"}}]]}]""")
      .setAttributeColumns(Map("TEXT" -> "token", "POS" -> "pos", "NER" -> "ner"))

    val matches = collectMatches(matcher.fit(dataset).transform(dataset))

    assert(matches.isEmpty)
  }

  it should "not match EXISTS false predicates when the attribute is present" taggedAs FastTest in {
    val dataset = simpleDataset("dogs bark", Seq("NOUN", "VERB"))
    val matcher = new RuleBasedMatcher()
      .setInputCols("sentence", "token", "pos")
      .setOutputCol("matches")
      .setRules("""[{"id":"present_exists_false","patterns":[[{"POS":{"EXISTS":false}}]]}]""")
      .setAttributeColumns(Map("TEXT" -> "token", "POS" -> "pos"))

    val matches = collectMatches(matcher.fit(dataset).transform(dataset))

    assert(matches.isEmpty)
  }

  it should "resolve overlaps deterministically with longest and priority strategies" taggedAs FastTest in {
    val dataset = simpleDataset("red big car", Seq("ADJ", "ADJ", "NOUN"))
    val rules =
      """
        |[
        |  {"id":"short","priority":10,"patterns":[[{"LOWER":"red"},{"LOWER":"big"}]]},
        |  {"id":"long","priority":1,"patterns":[[{"LOWER":"red"},{"LOWER":"big"},{"LOWER":"car"}]]}
        |]
        |""".stripMargin

    val base = new RuleBasedMatcher()
      .setInputCols("sentence", "token")
      .setOutputCol("matches")
      .setRules(rules)
      .setAttributeColumns(Map("TEXT" -> "token", "LOWER" -> "token"))

    val all = collectMatches(base.setOverlapStrategy("ALL").fit(dataset).transform(dataset))
    val longest =
      collectMatches(base.setOverlapStrategy("LONGEST").fit(dataset).transform(dataset))
    val priority =
      collectMatches(base.setOverlapStrategy("PRIORITY_LONGEST").fit(dataset).transform(dataset))

    assert(all.map(_.metadata("rule")) == Seq("short", "long"))
    assert(longest.map(_.metadata("rule")) == Seq("long"))
    assert(priority.map(_.metadata("rule")) == Seq("short"))
  }

  it should "keep matches inside sentence and document boundaries" taggedAs FastTest in {
    val dataset = repeatedSentenceIndexDataset()
    val rules =
      """[{"id":"hello_world","patterns":[[{"LOWER":"hello"},{"LOWER":"world"}]]}]"""

    val matcher = new RuleBasedMatcher()
      .setInputCols("document", "sentence", "token")
      .setOutputCol("matches")
      .setRules(rules)
      .setAttributeColumns(Map("TEXT" -> "token", "LOWER" -> "token"))

    val matches = collectMatches(matcher.fit(dataset).transform(dataset))
    assert(matches.map(_.result) == Seq("hello world", "hello world"))
    assert(matches.map(_.metadata("sentenceTokenBegin")) == Seq("0", "0"))
    assert(matches.map(_.metadata("documentTokenBegin")) == Seq("0", "0"))
    assert(matches.map(_.metadata("documentKey")).distinct.length == 2)
  }

  it should "keep sentence-local and document-level token indexes distinct" taggedAs FastTest in {
    val dataset = multiSentenceDataset()
    val rules =
      """
        |[
        |  {"id":"inside_second_sentence","patterns":[[{"LOWER":"dogs"},{"LOWER":"bark"}]]},
        |  {"id":"cross_sentence","patterns":[[{"LOWER":"world"},{"LOWER":"dogs"}]]}
        |]
        |""".stripMargin

    val matcher = new RuleBasedMatcher()
      .setInputCols("document", "sentence", "token")
      .setOutputCol("matches")
      .setRules(rules)
      .setAttributeColumns(Map("TEXT" -> "token", "LOWER" -> "token"))

    val matches = collectMatches(matcher.fit(dataset).transform(dataset))
    assert(matches.map(_.metadata("rule")) == Seq("inside_second_sentence"))
    assert(matches.head.metadata("sentence") == "1")
    assert(matches.head.metadata("sentenceTokenBegin") == "0")
    assert(matches.head.metadata("sentenceTokenEnd") == "1")
    assert(matches.head.metadata("documentTokenBegin") == "2")
    assert(matches.head.metadata("documentTokenEnd") == "3")
    assert(matches.head.metadata("tokenBegin") == "0")
    assert(matches.head.metadata("tokenEnd") == "1")
  }

  it should "use strict alignment by default and positional alignment only when requested" taggedAs FastTest in {
    val dataset = misalignedPosDataset()
    val rules = """[{"id":"noun","patterns":[[{"POS":"NOUN"}]]}]"""

    val strictMatcher = new RuleBasedMatcher()
      .setInputCols("sentence", "token", "pos")
      .setOutputCol("matches")
      .setRules(rules)
      .setAttributeColumns(Map("TEXT" -> "token", "POS" -> "pos"))

    val positionalMatcher = new RuleBasedMatcher()
      .setInputCols("sentence", "token", "pos")
      .setOutputCol("matches")
      .setRules(rules)
      .setAttributeColumns(Map("TEXT" -> "token", "POS" -> "pos"))
      .setAlignmentMode("POSITIONAL")

    assert(collectMatches(strictMatcher.fit(dataset).transform(dataset)).isEmpty)
    assert(
      collectMatches(positionalMatcher.fit(dataset).transform(dataset)).map(_.result) == Seq(
        "Dogs"))
  }

  it should "read external JSONL rule resources" taggedAs FastTest in {
    val ruleFile = Files.createTempFile("rule-based-matcher", ".jsonl")
    Files.write(
      ruleFile,
      Seq(
        """{"id":"dog","patterns":[[{"LOWER":"dogs"}]]}""",
        """{"id":"bark","patterns":[[{"LOWER":"bark"}]]}""")
        .mkString("\n")
        .getBytes(StandardCharsets.UTF_8))
    try {
      val matcher = new RuleBasedMatcher()
        .setInputCols("sentence", "token")
        .setOutputCol("matches")
        .setRulesResource(ruleFile.toString, ReadAs.TEXT, Map("format" -> "text"))
        .setAttributeColumns(Map("TEXT" -> "token", "LOWER" -> "token"))

      val matches = collectMatches(
        matcher.fit(simpleDataset("dogs bark")).transform(simpleDataset("dogs bark")))
      assert(matches.map(_.metadata("rule")) == Seq("dog", "bark"))
    } finally Files.deleteIfExists(ruleFile)
  }

  it should "fail early for invalid mappings, unknown attributes, and missing token input" taggedAs FastTest in {
    val dataset = simpleDataset("dogs bark", Seq("NOUN", "VERB"))

    intercept[IllegalArgumentException] {
      new RuleBasedMatcher()
        .setInputCols("sentence", "pos")
        .setOutputCol("matches")
        .setRules("""[{"id":"x","patterns":[[{"POS":"NOUN"}]]}]""")
        .fit(dataset)
    }.getMessage.contains("TOKEN")

    intercept[IllegalArgumentException] {
      new RuleBasedMatcher()
        .setInputCols("sentence", "token", "pos")
        .setOutputCol("matches")
        .setRules("""[{"id":"x","patterns":[[{"POSS":"NOUN"}]]}]""")
        .fit(dataset)
    }.getMessage.contains("unknown attributes")

    intercept[IllegalArgumentException] {
      new RuleBasedMatcher()
        .setInputCols("sentence", "token", "pos")
        .setOutputCol("matches")
        .setRules("""[{"id":"x","patterns":[[{"LEMMA":"dog"}]]}]""")
        .setAttributeColumns(Map("LEMMA" -> "pos"))
        .fit(dataset)
    }.getMessage.contains("expected one of token")
  }

  it should "fail early for malformed rules, invalid regex, invalid quantifiers, and unsafe patterns" taggedAs FastTest in {
    val dataset = simpleDataset("dogs bark", Seq("NOUN", "VERB"))

    Seq(
      """[{"id":"bad_operator","patterns":[[{"TEXT":{"ENDS_WITH":"s"}}]]}]""",
      """[{"id":"bad_quantifier","patterns":[[{"TEXT":"dogs","OP":"{3,1}"}]]}]""",
      """[{"id":"bad_regex","patterns":[[{"TEXT":{"REGEX":"["}}]]}]""",
      """[{"id":"bad_wildcards","patterns":[[{"OP":"*"},{"OP":"+"}]]}]""").foreach { rules =>
      val error = intercept[IllegalArgumentException] {
        new RuleBasedMatcher()
          .setInputCols("sentence", "token")
          .setOutputCol("matches")
          .setRules(rules)
          .setAttributeColumns(Map("TEXT" -> "token"))
          .fit(dataset)
      }
      assert(error.getMessage.contains("RuleBasedMatcher"))
    }
  }

  it should "reject unsafe direct flattened same-type annotations and use source metadata when present" taggedAs FastTest in {
    val text = "the dogs bark"
    val sentence = Seq(Annotation(DOCUMENT, 0, 12, text, Map("sentence" -> "0")))
    val token = Seq(
      Annotation(TOKEN, 0, 2, "the", Map("sentence" -> "0")),
      Annotation(TOKEN, 4, 7, "dogs", Map("sentence" -> "0")),
      Annotation(TOKEN, 9, 12, "bark", Map("sentence" -> "0")))
    val clean = Seq(
      Annotation(TOKEN, 4, 7, "dogs", Map("sentence" -> "0")),
      Annotation(TOKEN, 9, 12, "bark", Map("sentence" -> "0")))
    val dataset = withAnnotationTypes(
      Seq((sentence, token, clean)).toDF("sentence", "token", "clean"),
      Map("sentence" -> DOCUMENT, "token" -> TOKEN, "clean" -> TOKEN))
    val model = new RuleBasedMatcher()
      .setInputCols("sentence", "token", "clean")
      .setOutputCol("matches")
      .setRules("""[{"id":"removed","patterns":[[{"TEXT":"the","CLEAN":{"EXISTS":false}}]]}]""")
      .setAttributeColumns(Map("TEXT" -> "token", "CLEAN" -> "clean"))
      .fit(dataset)

    val error = intercept[IllegalArgumentException] {
      model.annotate(sentence ++ token ++ clean)
    }
    assert(error.getMessage.contains("cannot safely infer source columns"))

    val tagged = (sentence.map(tag(_, "sentence")) ++ token.map(tag(_, "token")) ++ clean.map(
      tag(_, "clean")))
    val matches = model.annotate(tagged)
    assert(matches.map(_.result) == Seq("the"))
  }

  it should "work after model and PipelineModel save/load" taggedAs FastTest in {
    val data = Seq("dogs bark").toDF("text")
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
    val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")
    val matcher = new RuleBasedMatcher()
      .setInputCols("document", "sentence", "token")
      .setOutputCol("matches")
      .setRules("""[{"id":"dogs","patterns":[[{"LOWER":"dogs"}]]}]""")
      .setAttributeColumns(Map("TEXT" -> "token", "LOWER" -> "token"))

    val pipelineModel =
      new Pipeline().setStages(Array(document, sentence, tokenizer, matcher)).fit(data)
    val beforeSave = collectMatches(pipelineModel.transform(data)).map(_.result)

    val pipelinePath = Files.createTempDirectory("rule-based-matcher-pipeline").toFile
    try {
      pipelineModel.write.overwrite().save(pipelinePath.getAbsolutePath)
      val loaded = PipelineModel.load(pipelinePath.getAbsolutePath)
      val afterSave = collectMatches(loaded.transform(data)).map(_.result)
      assert(beforeSave == afterSave)
    } finally new Directory(pipelinePath).deleteRecursively()
  }

  it should "work after direct model save/load" taggedAs FastTest in {
    val dataset = simpleDataset("dogs bark", Seq("NOUN", "VERB"))
    val model = new RuleBasedMatcher()
      .setInputCols("sentence", "token", "pos")
      .setOutputCol("matches")
      .setRules("""[{"id":"dogs","patterns":[[{"LOWER":"dogs","POS":"NOUN"}]]}]""")
      .setAttributeColumns(Map("TEXT" -> "token", "LOWER" -> "token", "POS" -> "pos"))
      .fit(dataset)

    val path = Files.createTempDirectory("rule-based-matcher-model").toFile
    try {
      model.write.overwrite().save(path.getAbsolutePath)
      val loaded = RuleBasedMatcherModel.load(path.getAbsolutePath)
      assert(collectMatches(loaded.transform(dataset)).map(_.result) == Seq("dogs"))
    } finally new Directory(path).deleteRecursively()
  }

  it should "work in LightPipeline without unsafe same-type splitting" taggedAs FastTest in {
    val data = Seq("the dogs bark").toDF("text")
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
    val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")
    val cleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("clean")
      .setStopWords(Array("the"))
    val matcher = new RuleBasedMatcher()
      .setInputCols("document", "sentence", "token", "clean")
      .setOutputCol("matches")
      .setRules("""[{"id":"removed","patterns":[[{"TEXT":"the","CLEAN":{"EXISTS":false}}]]}]""")
      .setAttributeColumns(Map("TEXT" -> "token", "CLEAN" -> "clean"))

    val model =
      new Pipeline().setStages(Array(document, sentence, tokenizer, cleaner, matcher)).fit(data)
    val light = new LightPipeline(model)
    val matches = light
      .fullAnnotate("the dogs bark")("matches")
      .map(_.asInstanceOf[Annotation])
    assert(matches.map(_.result) == Seq("the"))
  }

  it should "handle empty token inputs without matches" taggedAs FastTest in {
    val dataset = withAnnotationTypes(
      Seq((Seq(Annotation(DOCUMENT, 0, 0, "", Map("sentence" -> "0"))), Seq.empty[Annotation]))
        .toDF("sentence", "token"),
      Map("sentence" -> DOCUMENT, "token" -> TOKEN))
    val matcher = new RuleBasedMatcher()
      .setInputCols("sentence", "token")
      .setOutputCol("matches")
      .setRules("""[{"id":"anything","patterns":[[{}]]}]""")
      .setAttributeColumns(Map("TEXT" -> "token"))
    assert(collectMatches(matcher.fit(dataset).transform(dataset)).isEmpty)
  }

  it should "resolve document keys by original annotation index" taggedAs FastTest in {
    val text = "dogs bark"
    val mixedAnnotations = Seq(
      Annotation(TOKEN, 0, 3, "dogs", Map("sentence" -> "0")),
      Annotation(DOCUMENT, 0, 8, text, Map("sentence" -> "0")))
    val annotationsByColumn = Map("mixed_doc" -> mixedAnnotations)

    val resolved =
      invokeDocumentByKey("mixed_doc:1:0:8:0", annotationsByColumn)

    assert(resolved.map(_.result).contains(text))
  }

  it should "reconstruct chunk text from the original document index in Spark pipelines" taggedAs FastTest in {
    val text = "Dr. Smith"
    val mixedDocument = Seq(
      Annotation(TOKEN, 0, 1, "Dr", Map("sentence" -> "0")),
      Annotation(DOCUMENT, 0, 8, text, Map("sentence" -> "0")),
      Annotation(DOCUMENT, 0, 8, "Bad Smith", Map("sentence" -> "0")))
    val tokens = Seq(
      Annotation(TOKEN, 0, 1, "Dr", Map("sentence" -> "0")),
      Annotation(TOKEN, 2, 2, ".", Map("sentence" -> "0")),
      Annotation(TOKEN, 4, 8, "Smith", Map("sentence" -> "0")))
    val dataset = withAnnotationTypes(
      Seq((mixedDocument, tokens)).toDF("mixed_doc", "token"),
      Map("mixed_doc" -> DOCUMENT, "token" -> TOKEN))
    val matcher = new RuleBasedMatcher()
      .setInputCols("mixed_doc", "token")
      .setOutputCol("matches")
      .setRules(
        """[{"id":"honorific_name","patterns":[[{"TEXT":"Dr"},{"TEXT":"."},{"TEXT":"Smith"}]]}]""")
      .setAttributeColumns(Map("TEXT" -> "token"))

    val matches = collectMatches(matcher.fit(dataset).transform(dataset))

    assert(matches.map(_.result) == Seq(text))
    assert(matches.head.begin == 0)
    assert(matches.head.end == 8)
    assert(matches.head.metadata("documentKey").startsWith("mixed_doc:1:"))
  }

  it should "use covering document text before falling back to token-joined chunks" taggedAs FastTest in {
    val document = Annotation(DOCUMENT, 0, 8, "Dr. Smith", Map("sentence" -> "1"))
    val candidate: MatchCandidate = MatchCandidate(
      ruleId = "name",
      label = "NAME",
      priority = 0,
      patternIndex = 0,
      ruleIndex = 0,
      documentKey = "no_document:token:0",
      sentence = "0",
      sentenceTokenStart = 0,
      sentenceTokenEnd = 2,
      documentTokenStart = 0,
      documentTokenEnd = 2,
      begin = 0,
      end = 8,
      result = "Dr . Smith")

    val chunkText = invokeChunkResult(candidate, Map.empty, Seq(document))

    assert(chunkText == "Dr. Smith")
  }

  private def addressRules(attribute: String): String =
    s"""
       |[{
       |  "id": "address_like_location",
       |  "label": "ADDRESS",
       |  "priority": 10,
       |  "patterns": [[
       |    {"POS": "NUM"},
       |    {"POS": "NUM"},
       |    {"POS": "NOUN"},
       |    {"$attribute": "GPE", "OP": "+"}
       |  ]]
       |}]
       |""".stripMargin

  private def singleTokenRule(
      id: String,
      label: String,
      attribute: String,
      value: String): String =
    s"""[{"id":"$id","label":"$label","patterns":[[{"$attribute":"$value"}]]}]"""

  private def simpleDataset(
      text: String,
      posTags: Seq[String] = Seq.empty,
      columnPrefix: String = ""): DataFrame = {
    val tokenTexts = text.split(" ").filter(_.nonEmpty).toSeq
    var cursor = 0
    val tokens = tokenTexts.map { token =>
      val begin = text.indexOf(token, cursor)
      val end = begin + token.length - 1
      cursor = end + 1
      Annotation(TOKEN, begin, end, token, Map("sentence" -> "0"))
    }
    val pos = tokens.zip(posTags).map { case (token, pos) =>
      Annotation(POS, token.begin, token.end, pos, Map("sentence" -> "0"))
    }
    val sentence = Seq(
      Annotation(DOCUMENT, 0, Math.max(0, text.length - 1), text, Map("sentence" -> "0")))
    val base = Seq((sentence, tokens, pos)).toDF(
      s"${columnPrefix}sentence",
      s"${columnPrefix}token",
      s"${columnPrefix}pos")
    withAnnotationTypes(
      base,
      Map(
        s"${columnPrefix}sentence" -> DOCUMENT,
        s"${columnPrefix}token" -> TOKEN,
        s"${columnPrefix}pos" -> POS))
  }

  private def addressDataset(): DataFrame = {
    val text = "443 8th Street New York"
    val sentence = Seq(Annotation(DOCUMENT, 0, 22, text, Map("sentence" -> "0")))
    val token = Seq(
      Annotation(TOKEN, 0, 2, "443", Map("sentence" -> "0")),
      Annotation(TOKEN, 4, 6, "8th", Map("sentence" -> "0")),
      Annotation(TOKEN, 8, 13, "Street", Map("sentence" -> "0")),
      Annotation(TOKEN, 15, 17, "New", Map("sentence" -> "0")),
      Annotation(TOKEN, 19, 22, "York", Map("sentence" -> "0")))
    val pos = Seq(
      Annotation(POS, 0, 2, "NUM", Map("sentence" -> "0")),
      Annotation(POS, 4, 6, "NUM", Map("sentence" -> "0")),
      Annotation(POS, 8, 13, "NOUN", Map("sentence" -> "0")),
      Annotation(POS, 15, 17, "PROPN", Map("sentence" -> "0")),
      Annotation(POS, 19, 22, "PROPN", Map("sentence" -> "0")))
    val ner = Seq(
      Annotation(NAMED_ENTITY, 0, 2, "O", Map("sentence" -> "0")),
      Annotation(NAMED_ENTITY, 4, 6, "O", Map("sentence" -> "0")),
      Annotation(NAMED_ENTITY, 8, 13, "O", Map("sentence" -> "0")),
      Annotation(NAMED_ENTITY, 15, 17, "B-GPE", Map("sentence" -> "0")),
      Annotation(NAMED_ENTITY, 19, 22, "I-GPE", Map("sentence" -> "0")))
    withAnnotationTypes(
      Seq((sentence, token, pos, ner)).toDF("sentence", "token", "pos", "ner"),
      Map("sentence" -> DOCUMENT, "token" -> TOKEN, "pos" -> POS, "ner" -> NAMED_ENTITY))
  }

  private def datasetWithEmptyNer(text: String, posTags: Seq[String]): DataFrame = {
    val tokenTexts = text.split(" ").filter(_.nonEmpty).toSeq
    var cursor = 0
    val tokens = tokenTexts.map { token =>
      val begin = text.indexOf(token, cursor)
      val end = begin + token.length - 1
      cursor = end + 1
      Annotation(TOKEN, begin, end, token, Map("sentence" -> "0"))
    }
    val pos = tokens.zip(posTags).map { case (token, pos) =>
      Annotation(POS, token.begin, token.end, pos, Map("sentence" -> "0"))
    }
    val sentence =
      Seq(Annotation(DOCUMENT, 0, Math.max(0, text.length - 1), text, Map("sentence" -> "0")))
    withAnnotationTypes(
      Seq((sentence, tokens, pos, Seq.empty[Annotation])).toDF("sentence", "token", "pos", "ner"),
      Map("sentence" -> DOCUMENT, "token" -> TOKEN, "pos" -> POS, "ner" -> NAMED_ENTITY))
  }

  private def lemmaDataset(): DataFrame = {
    val text = "Dogs bark"
    val sentence = Seq(Annotation(DOCUMENT, 0, 8, text, Map("sentence" -> "0")))
    val token = Seq(
      Annotation(TOKEN, 0, 3, "Dogs", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 8, "bark", Map("sentence" -> "0")))
    val lemma = Seq(
      Annotation(TOKEN, 0, 3, "dog", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 8, "bark", Map("sentence" -> "0")))
    val pos = Seq(
      Annotation(POS, 0, 3, "NOUN", Map("sentence" -> "0")),
      Annotation(POS, 5, 8, "VERB", Map("sentence" -> "0")))
    withAnnotationTypes(
      Seq((sentence, token, lemma, pos)).toDF("sentence", "token", "lemma", "pos"),
      Map("sentence" -> DOCUMENT, "token" -> TOKEN, "lemma" -> TOKEN, "pos" -> POS))
  }

  private def multiSentenceDataset(): DataFrame = {
    val text = "Hello world. Dogs bark."
    val document = Seq(Annotation(DOCUMENT, 0, 22, text, Map("sentence" -> "0")))
    val sentence = Seq(
      Annotation(DOCUMENT, 0, 11, "Hello world.", Map("sentence" -> "0")),
      Annotation(DOCUMENT, 13, 22, "Dogs bark.", Map("sentence" -> "1")))
    val token = Seq(
      Annotation(TOKEN, 0, 4, "Hello", Map("sentence" -> "0")),
      Annotation(TOKEN, 6, 10, "world", Map("sentence" -> "0")),
      Annotation(TOKEN, 13, 16, "Dogs", Map("sentence" -> "1")),
      Annotation(TOKEN, 18, 21, "bark", Map("sentence" -> "1")))
    withAnnotationTypes(
      Seq((document, sentence, token)).toDF("document", "sentence", "token"),
      Map("document" -> DOCUMENT, "sentence" -> DOCUMENT, "token" -> TOKEN))
  }

  private def misalignedPosDataset(): DataFrame = {
    val text = "Dogs bark"
    val sentence = Seq(Annotation(DOCUMENT, 0, 8, text, Map("sentence" -> "0")))
    val token = Seq(
      Annotation(TOKEN, 0, 3, "Dogs", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 8, "bark", Map("sentence" -> "0")))
    val pos = Seq(
      Annotation(POS, 1, 4, "NOUN", Map("sentence" -> "0")),
      Annotation(POS, 6, 9, "VERB", Map("sentence" -> "0")))
    withAnnotationTypes(
      Seq((sentence, token, pos)).toDF("sentence", "token", "pos"),
      Map("sentence" -> DOCUMENT, "token" -> TOKEN, "pos" -> POS))
  }

  private def repeatedSentenceIndexDataset(): DataFrame = {
    val doc1 = "hello world"
    val doc2 = "hello world"
    val document = Seq(
      Annotation(DOCUMENT, 0, 10, doc1, Map("sentence" -> "0")),
      Annotation(DOCUMENT, 0, 10, doc2, Map("sentence" -> "0")))
    val sentence = Seq(
      Annotation(DOCUMENT, 0, 10, doc1, Map("sentence" -> "0")),
      Annotation(DOCUMENT, 0, 10, doc2, Map("sentence" -> "0")))
    val token = Seq(
      Annotation(TOKEN, 0, 4, "hello", Map("sentence" -> "0")),
      Annotation(TOKEN, 6, 10, "world", Map("sentence" -> "0")),
      Annotation(TOKEN, 0, 4, "hello", Map("sentence" -> "0")),
      Annotation(TOKEN, 6, 10, "world", Map("sentence" -> "0")))
    val taggedDocument = document.zipWithIndex.map { case (ann, idx) =>
      tag(ann, s"document_$idx")
    }
    val taggedSentence = sentence.zipWithIndex.map { case (ann, idx) =>
      tag(ann, s"sentence_$idx")
    }
    val taggedToken = token.zipWithIndex.map { case (ann, idx) =>
      tag(ann, if (idx < 2) "token_0" else "token_1")
    }
    withAnnotationTypes(
      Seq((taggedDocument, taggedSentence, taggedToken)).toDF("document", "sentence", "token"),
      Map("document" -> DOCUMENT, "sentence" -> DOCUMENT, "token" -> TOKEN))
  }

  private def collectMatches(dataset: DataFrame): Seq[Annotation] =
    Annotation.collect(dataset, "matches").flatten.toSeq

  private def invokeDocumentByKey(
      documentKey: String,
      annotationsByColumn: Map[String, Seq[Annotation]]): Option[Annotation] = {
    val method = classOf[RuleBasedMatcherModel].getDeclaredMethods
      .find(method => method.getName == "documentByKey" && method.getParameterCount == 2)
      .get
    method.setAccessible(true)
    method
      .invoke(new RuleBasedMatcherModel(), documentKey, annotationsByColumn)
      .asInstanceOf[Option[Annotation]]
  }

  private def invokeChunkResult(
      candidate: MatchCandidate,
      annotationsByColumn: Map[String, Seq[Annotation]],
      documents: Seq[Annotation]): String = {
    val method = classOf[RuleBasedMatcherModel].getDeclaredMethods
      .find(method => method.getName == "chunkResult" && method.getParameterCount == 3)
      .get
    method.setAccessible(true)
    method
      .invoke(new RuleBasedMatcherModel(), candidate, annotationsByColumn, documents)
      .asInstanceOf[String]
  }

  private def tag(annotation: Annotation, sourceColumn: String): Annotation =
    annotation.copy(metadata = annotation.metadata + ("source_column" -> sourceColumn))

  private def withAnnotationTypes(
      dataset: DataFrame,
      annotatorTypes: Map[String, String]): DataFrame =
    annotatorTypes.foldLeft(dataset) { case (df, (columnName, annotatorType)) =>
      val metadata = new MetadataBuilder().putString("annotatorType", annotatorType).build()
      df.withColumn(columnName, col(columnName).as(columnName, metadata))
    }
}
