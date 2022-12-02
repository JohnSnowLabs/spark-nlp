/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.spell.context

import com.johnsnowlabs.nlp.SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp.annotator.RecursiveTokenizer
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.{PrefixedToken, SuffixedToken}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.spell.context.parser._
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, LightPipeline, SparkAccessor}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.junit.Assert.assertEquals
import org.scalatest.flatspec.AnyFlatSpec

import java.io._
import scala.collection.JavaConverters._

class ContextSpellCheckerTestSpec extends AnyFlatSpec {

  trait Scope extends WeightedLevenshtein {
    val weights: Map[String, Map[String, Float]] =
      Map("1" -> Map("l" -> 0.5f), "!" -> Map("l" -> 0.4f), "F" -> Map("P" -> 0.2f))
  }

  trait distFile extends WeightedLevenshtein {
    val weights: Map[String, Map[String, Float]] = loadWeights("src/test/resources/dist.psv")
  }
  // This test fails in GitHub Actions
  "Spell Checker" should "provide appropriate scores - sentence level" taggedAs SlowTest in {

    def time[R](block: => R): R = {
      val t0 = System.nanoTime()
      val result = block // call-by-name
      val t1 = System.nanoTime()
      println("Elapsed time: " + (t1 - t0) + "ns")
      result
    }

    val data = Seq("This is a correct sentence .", "This is a correct bananas .").toDF("text")

    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("doc")

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("doc"))
      .setOutputCol("token")

    val spellChecker = ContextSpellCheckerModel
      .pretrained()
      .setTradeOff(12.0f)
      .setInputCols("token")
      .setOutputCol("checked")

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker)).fit(data)
    import com.johnsnowlabs.nlp.functions._
    val results = time {
      pipeline
        .transform(data)
        .select("checked")
        .mapAnnotationsCol[Option[String]](
          "checked",
          "checked",
          "language",
          (x: Seq[Annotation]) => x.head.metadata.get("cost"))
        .collect
        .map(_.getString(0).toDouble)
    }

    assert(results(0) < results(1))

  }
  "ContextSpellchker" should "return correct order" taggedAs SlowTest in new distFile {
    val data: DataFrame = Seq("It was a cold. The country was white withh snow .").toDF("text")
    val documentAssembler: DocumentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val sentenceDetector: SentenceDetector =
      new SentenceDetector().setInputCols("document").setOutputCol("sentences")
    val tokenizer: Tokenizer = new Tokenizer().setInputCols("sentences").setOutputCol("tokens")
    val spell_checker: ContextSpellCheckerModel = ContextSpellCheckerModel
      .pretrained()
      .setInputCols("tokens")
      .setOutputCol("corrected_tokens")
    val pipeline: PipelineModel = new Pipeline()
      .setStages(Array(documentAssembler, sentenceDetector, tokenizer, spell_checker))
      .fit(data)

    val output_df: DataFrame = pipeline.transform(data)

    val annotation: Array[Annotation] = Annotation.collect(output_df, "corrected_tokens").flatten
    assertEquals("It", annotation.head.result)
    assertEquals("was", annotation(1).result)
    assertEquals("a", annotation(2).result)
    assertEquals("cold", annotation(3).result)
    assertEquals(".", annotation(4).result)
    assertEquals("The", annotation(5).result)
    assertEquals("country", annotation(6).result)
    assertEquals("was", annotation(7).result)
    assertEquals("white", annotation(8).result)
    assertEquals("with", annotation(9).result)
    assertEquals("snow", annotation(10).result)

  }

  // This test fails in GitHub Actions
  "Special classes" should "serialize/deserialize properly during model save" taggedAs SlowTest in {
    import SparkAccessor.spark

    val specialClasses = Seq(
      new AgeToken,
      new UnitToken,
      new NumberToken,
      new LocationClass("./src/test/resources/spell/locations.txt"),
      new NamesClass("./src/test/resources/spell/names.txt"),
      new MedicationClass("./src/test/resources/spell/meds.txt"),
      new DateToken)

    specialClasses.foreach { specialClass =>
      val dataPathObject = "/tmp/object"

      val f = new File(dataPathObject)
      if (f.exists()) FileUtils.deleteDirectory(f)

      // persist object
      FileUtils.deleteDirectory(new File(dataPathObject))
      spark.sparkContext.parallelize(Seq(specialClass)).saveAsObjectFile(dataPathObject)

      // load object
      val sc = spark.sparkContext.objectFile[SpecialClassParser](dataPathObject).collect().head
      assert(sc.transducer != null)
      sc match {
        case vp: VocabParser => assert(vp.vocab != null)
        case _ =>
      }

      sc.transducer.transduce("aaa")
    }
  }

  "Special classes" should "serialize/deserialize properly - during execution" taggedAs SlowTest in {

    val specialClasses = Seq(
      new AgeToken,
      new UnitToken,
      new NumberToken,
      new LocationClass("./src/test/resources/spell/locations.txt"),
      new NamesClass("./src/test/resources/spell/names.txt"),
      new MedicationClass("./src/test/resources/spell/meds.txt"),
      new DateToken)

    specialClasses.foreach { specialClass =>
      val path = "special_class.ser"
      val f = new File(path)
      if (f.exists()) FileUtils.forceDelete(f)

      // write to disk
      val fileOut: FileOutputStream = new FileOutputStream(path)
      val out: ObjectOutputStream = new ObjectOutputStream(fileOut)

      out.writeObject(specialClass)
      out.close()

      // read from disk
      val fileIn: FileInputStream = new FileInputStream(path)
      val in: ObjectInputStream = new ObjectInputStream(fileIn)
      val deserialized = in.readObject.asInstanceOf[SpecialClassParser]
      assert(deserialized.transducer != null)
      deserialized match {
        case vp: VocabParser =>
          assert(vp.vocab != null)
        case _ =>
      }
      deserialized.transducer.transduce("something")
      in.close()
    }
  }

  "weighted Levenshtein distance" should "work from file" taggedAs SlowTest in new distFile {
    assert(wLevenshteinDist("water", "Water", weights) < 1.0f)
    assert(wLevenshteinDist("50,000", "50,C00", weights) < 1.0f)
  }

  "weighted Levenshtein distance" should "produce weighted results" taggedAs SlowTest in new Scope {
    assert(
      wLevenshteinDist("clean", "c1ean", weights) > wLevenshteinDist("clean", "c!ean", weights))
    assert(
      wLevenshteinDist("clean", "crean", weights) > wLevenshteinDist("clean", "c!ean", weights))
    assert(
      wLevenshteinDist("Patient", "Fatient", weights) < wLevenshteinDist(
        "Patient",
        "Aatient",
        weights))
  }

  "weighted Levenshtein distance" should "handle insertions and deletions" taggedAs SlowTest in new Scope {
    override val weights: Map[String, Map[String, Float]] =
      loadWeights("src/test/resources/distance.psv")

    val cost1: Float = weights("F")("P") + weights("a")("e")
    assert(wLevenshteinDist("Procedure", "Frocedura", weights) == cost1)

    val cost2: Float = weights("v")("y") + weights("iƐ")("if")
    assert(wLevenshteinDist("qualifying", "qualiving", weights) == cost2)

    val cost3: Float = weights("a")("o") + weights("^Ɛ")("^t")
    assert(wLevenshteinDist("to", "a", weights) == cost3)
  }

  "a Spell Checker" should "correctly preprocess training data" taggedAs FastTest in {

    val path = "src/test/resources/test.txt"
    val dataset = SparkAccessor.spark.sparkContext.textFile(path).toDF("text")

    val assembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("doc")

    val tokenizer: RecursiveTokenizer = new RecursiveTokenizer()
      .setInputCols(Array("doc"))
      .setOutputCol("token")

    val spellChecker = new ContextSpellCheckerApproach().setMinCount(1.0)

    val stages = Array(assembler, tokenizer)

    val trainingPipeline = new Pipeline()
      .setStages(stages)
      .fit(dataset)

    val (map, classes) = spellChecker.genVocab(trainingPipeline.transform(dataset))
    assert(map.exists(_._1.equals("“seed”")))

    val totalTokenCount = 35
    assert(map.size == 23)
    assert(
      map.getOrElse("_EOS_", 0.0) == math.log(2.0) - math.log(totalTokenCount),
      "Two sentences should cause two _BOS_ markers")
    assert(
      map.getOrElse("_BOS_", 0.0) == math.log(2.0) - math.log(totalTokenCount),
      "Two sentences should cause two _EOS_ markers")

    assert(classes.size == 23, "")

  }

  "a Spell Checker" should "work in a pipeline with Tokenizer" taggedAs SlowTest in {
    val data = Seq(
      "It was a cold , dreary day and the country was white with smow .",
      "He wos re1uctant to clange .",
      "he is gane .").toDF("text")

    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("doc")

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("doc"))
      .setOutputCol("token")

    val spellChecker = ContextSpellCheckerModel
      .pretrained()
      .setTradeOff(12.0f)
      .setInputCols("token")
      .setOutputCol("checked")

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker)).fit(data)
    pipeline.transform(data).select("checked").show(truncate = false)

  }

  "a Spell Checker" should "work in a light pipeline" taggedAs SlowTest in {
    import SparkAccessor.spark
    import spark.implicits._

    val data =
      Array("Yesterday I lost my blue unikorn .", "Through a note of introduction from Bettina.")

    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("doc")

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("doc"))
      .setOutputCol("token")

    val spellChecker = ContextSpellCheckerModel
      .pretrained()
      .setTradeOff(12.0f)
      .setInputCols("token")
      .setOutputCol("checked")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, spellChecker))
      .fit(Seq.empty[String].toDF("text"))
    val lp = new LightPipeline(pipeline)
    lp.annotate(data ++ data ++ data)
  }

  "a Spell Checker" should "correctly handle paragraphs defined by newlines" taggedAs SlowTest in {
    import SparkAccessor.spark
    import spark.implicits._

    val data = Seq(
      "Incruse Ellipta, 1 PUFF, Inhalation,\nQAM\n\nlevothyroxine 50 meg (0.05 mg) oral\ntablet, See Instructions\n\nlisinopril 20 mg oral tablet, See\nInstructions, 5 refills\n\nloratadine 10 mg oral tablet, 10 MG=\n1 TAB, PO, Dally\n\nPercocet 10/325 oral tablet, 2 TAB,\nPO, TID, PRN")
      .toDF("text")

    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("doc")

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("doc"))
      .setOutputCol("token")
      .setTargetPattern("[a-zA-Z0-9]+|\n|\n\n|\\(|\\)|\\.|\\,")

    val spellChecker = ContextSpellCheckerModel
      .pretrained()
      .setTradeOff(12.0f)
      .setInputCols("token")
      .setOutputCol("checked")
      .setUseNewLines(true)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker)).fit(data)
    pipeline.transform(data).select("checked").show(truncate = false)

  }

  "a Spell Checker" should "correctly handle multiple sentences" taggedAs SlowTest in {

    import SparkAccessor.spark
    import spark.implicits._

    val data = Seq(
      "It had been raining just this way all day and hal1 of last night, and to all" +
        " appearances it intended to continue raining in the same manner for another twenty-four hours." +
        " Yesterday the Yard had ben a foot deep in nice clean snow, the result of the blizzard that had" +
        " sweptr over Wisining and New Eng1and in general two days before.").toDF("text")

    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("doc")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("doc"))
      .setOutputCol("sentence")

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val spellChecker = ContextSpellCheckerModel
      .pretrained()
      .setInputCols("token")
      .setOutputCol("checked")
      .setUseNewLines(true)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceDetector, tokenizer, spellChecker))
      .fit(data)
    val result = pipeline.transform(data)
    val checked = result.select("checked").as[Array[Annotation]].collect
    val firstSent = checked.head.filter(_.metadata("sentence") == "0").map(_.result)
    val secondSent = checked.head.filter(_.metadata("sentence") == "1").map(_.result)

    assert(firstSent.contains("half"))
    assert(secondSent.contains("swept"))
  }

  "a model" should "correctly update word classes" taggedAs SlowTest in {

    import SparkAccessor.spark
    import spark.implicits._

    val data =
      Seq("We should take a trup to Supercalifragilisticexpialidoccious Land").toDF("text")
    val meds: java.util.ArrayList[String] = new java.util.ArrayList[String]()
    meds.add("Supercalifragilisticexpialidocious")

    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("doc")

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("doc"))
      .setOutputCol("token")

    val spellChecker = ContextSpellCheckerModel
      .pretrained()
      .updateVocabClass("_LOC_", meds, append = false)
      .setInputCols("token")
      .setOutputCol("checked")
      .setUseNewLines(true)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker)).fit(data)
    val result = pipeline.transform(data)
    val checked = result.select("checked").as[Array[Annotation]].collect
    // check the spell checker was able to correct the word according to the update in the class
    pipeline.stages.last
      .asInstanceOf[ContextSpellCheckerModel]
      .write
      .overwrite
      .save("./test_spell_checker")
    assert(checked.head.map(_.result).contains("Supercalifragilisticexpialidocious"))

  }

  "a model" should "serialize properly" taggedAs SlowTest in {

    val ocrSpellModel = ContextSpellCheckerModel.pretrained()
    assert(
      ocrSpellModel.specialTransducers.getOrDefault.size == 4,
      "default pretrained should come with 4 classes")

    // now we update the classes, and persist/unpersist the model
    ocrSpellModel.setSpecialClassesTransducers(Seq(new DateToken, new NumberToken))
    ocrSpellModel.write.overwrite.save("./test_spell_checker")
    val loadedModel = ContextSpellCheckerModel.read.load("./test_spell_checker")

    // cope with potential change in element order in list
    val sortedTransducers = loadedModel.specialTransducers.getOrDefault.sortBy(_.label)

    assert(sortedTransducers.head.label == "_DATE_")
    assert(
      sortedTransducers.head.generateTransducer
        .transduce("10710/2018", 1)
        .asScala
        .map(_.term())
        .toSeq
        .contains("10/10/2018"))

    assert(sortedTransducers(1).label == "_NUM_")
    assert(
      sortedTransducers(1).generateTransducer
        .transduce("50,C00", 1)
        .asScala
        .map(_.term())
        .toSeq
        .contains("50,000"))

    val trellis = Array(
      Array.fill(6)(("the", 0.8, "the")),
      Array.fill(6)(("end", 1.2, "end")),
      Array.fill(6)((".", 1.2, ".")))
    val (decoded, _) = loadedModel.decodeViterbi(trellis)
    assert(decoded.deep.equals(Array("the", "end", ".").deep))

  }

  "number classes" should "recognize different number patterns" taggedAs FastTest in {
    val number = new NumberToken
    val transducer = number.generateTransducer

    assert(transducer.transduce("100.3").asScala.toList.exists(_.distance == 0))
    assert(number.separate("$40,000").equals(number.label))
  }

  "date classes" should "recognize different date and time formats" taggedAs FastTest in {
    val date = new DateToken
    val transducer = date.generateTransducer

    assert(transducer.transduce("10/25/1982").asScala.toList.exists(_.distance == 0))
    assert(date.separate("10/25/1982").equals(date.label))
  }

  "suffixes and prefixes" should "recognized and handled properly" taggedAs FastTest in {
    val suffixedToken = SuffixedToken(Array(")", ","))
    val prefixedToken = PrefixedToken(Array("("))

    var tmp = suffixedToken.separate("People,")
    assert(tmp.equals("People ,"))

    tmp = prefixedToken.separate(suffixedToken.separate("(08/10/1982)"))
    assert(tmp.equals("( 08/10/1982 )"))
  }

  "when using ContextSpellchecker" should "Adding Multiple values for updateVocabClass when append=true should not crash" taggedAs SlowTest in {

    import SparkAccessor.spark
    import spark.implicits._

    val data =
      Seq("We should take a trup to Supercalifragilisticexpialidoccious Land").toDF("text")
    val meds: java.util.ArrayList[String] = new java.util.ArrayList[String]()
    meds.add("Supercalifragilisticexpialidocious")
    meds.add("Monika")
    meds.add("Peter")

    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("doc")

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("doc"))
      .setOutputCol("token")

    val spellChecker = ContextSpellCheckerModel
      .pretrained()
      .updateVocabClass("_LOC_", meds, append = true)
      .setInputCols("token")
      .setOutputCol("checked")
      .setUseNewLines(true)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker)).fit(data)
    val result = pipeline.transform(data).show()

  }

}
