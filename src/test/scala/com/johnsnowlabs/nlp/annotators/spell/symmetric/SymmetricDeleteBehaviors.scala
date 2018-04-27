package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp._

import org.scalatest._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

import SparkAccessor.spark.implicits._


trait SymmetricDeleteBehaviors { this: FlatSpec =>

  val spellChecker = new SymmetricDeleteApproach()
      .setCorpus(ExternalResource("src/test/resources/spell/sherlockholmes.txt",
                 ReadAs.LINE_BY_LINE,
                 Map("tokenPattern" -> "[a-zA-Z]+")))
      .fit(DataBuilder.basicDataBuild("dummy"))

  def testSuggestions(): Unit = {
    // The code above should be executed first
    val silent = true
    spellChecker.getSuggestedCorrections("problex", silent)
    println("done")
  }

  def testLevenshteinDistance(): Unit = {
    printDistance("kitten", "sitting")
    printDistance("rosettacode", "raisethysword")
  }

  private def printDistance(s1:String, s2:String): Unit =
    println("%s -> %s : %d".format(s1, s2, spellChecker.levenshteinDistance(s1, s2)))

  def testSimpleCheck(wordAnswer: Seq[(String, String)]): Unit ={
    s"symspell checker " should s"successfully correct a misspell" in {
      val misspell = wordAnswer.head._1
      val correction = spellChecker.check(misspell)
      assert(correction == wordAnswer.head._2)
    }
  }

  def testSeveralChecks(wordAnswer:  Seq[(String, String)]): Unit = {
    s"symspell checker " should s"successfully correct several misspells" in {
      wordAnswer.foreach( wa => {
        val misspell = wa._1
        val correction = spellChecker.check(misspell)
        assert(correction == wa._2)
      })
    }
  }

  def testAccuracyChecks(wordAnswer: Seq[(String, String)]): Unit = {
    s"spell checker" should s" correct words with at least a fair accuracy" in {
      val result = wordAnswer.count(wa =>
        spellChecker.check(wa._1) == wa._2) / wordAnswer.length.toDouble
      println(result)
      assert(result > 0.60, s"because result: $result did was below: 0.60")
    }
  }


  def testBigPipeline(): Unit = {
    s"a SymSpellChecker annotator using a big pipeline" should "successfully correct words" in {
      val data = ContentProvider.parquetData.limit(3000)
      val corpusData = Seq.empty[String].toDS

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val spell = new SymmetricDeleteApproach()
        .setInputCols(Array("token"))
        .setOutputCol("spell")
        .setCorpus(ExternalResource("src/test/resources/spell/sherlockholmes.txt",
                                    ReadAs.LINE_BY_LINE,
                                    Map("tokenPattern" -> "[a-zA-Z]+")))

      val finisher = new Finisher()
        .setInputCols("spell")

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          spell,
          finisher
        ))

      val model = pipeline.fit(corpusData.select(corpusData.col("value").as("text")))
      model.transform(data).show(10, false)

    }
  }

  def testIndividualWords(): Unit = {
    s"a SymSpellChecker annotator with pipeline of individual words" should
      "successfully correct words with good accuracy" in {

      val corpusData = Seq.empty[String].toDS
      val path = "src/test/resources/spell/misspelled_words.csv"
      val data = SparkAccessor.spark.read.format("csv").option("header", "true").load(path)
      data.show(10)

      val documentAssembler = new DocumentAssembler()
        .setInputCol("misspell")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val corpusPath = "src/test/resources/spell/sherlockholmes.txt"
      // val corpusPath = "/home/danilo/IdeaProjects/spark-nlp-models/src/main/resources/spell/wiki1_en.txt"
      // val corpusPath = "/home/danilo/PycharmProjects/SymSpell/coca2017.txt"
      // val corpusPath = "/home/danilo/IdeaProjects/spark-nlp-models/src/main/resources/spell/coca2017/2017_spok.txt"

      val spell = new SymmetricDeleteApproach()
        .setInputCols(Array("token"))
        .setOutputCol("spell")
        .setCorpus(ExternalResource(corpusPath,
                                    ReadAs.LINE_BY_LINE,
                                    Map("tokenPattern" -> "[a-zA-Z]+")))

      val finisher = new Finisher()
        .setInputCols("spell")
        .setOutputAsArray(false)

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          spell,
          finisher
        ))

      val model = pipeline.fit(corpusData.select(corpusData.col("value").as("misspell")))
      var correctedData = model.transform(data)
      correctedData.show(10)
      correctedData = correctedData.withColumn("prediction",
                                               when(col("word") === col("finished_spell"), 1).otherwise(0))
      correctedData.show(10)
      val rightCorrections = correctedData.filter(col("prediction")===1).count()
      val wrongCorrections = correctedData.filter(col("prediction")===0).count()
      printf("Right Corrections: %d \n", rightCorrections)
      printf("Wrong Corrections: %d \n", wrongCorrections)
    }
  }

  def testSparkDataset(): Unit = {
    s"a SymSpellChecker annotator with pipeline of" should
      "successfully correct words" in {

      val corpusData = Seq.empty[String].toDS
      val path = "src/test/resources/spell/sherlockholmes.txt"
      var data = SparkAccessor.spark.read.format("csv").load(path)
      data = data.select(data.col("_c0").as("text"))
      data.show(10)

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val finisher = new Finisher()
        .setInputCols("document")
        .setOutputCols("finished")
        .setOutputAsArray(false)

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          finisher
        ))

      val model = pipeline.fit(corpusData.select(corpusData.col("value").as("text")))
      val transformation = model.transform(data)
      transformation.show(10, false)
      println("done")
      // val resourceList = transformation.select(transformation.col("finished"))
      val resourceList = transformation.select("finished").map(r => r.getString(0)).collect.toList

      println(resourceList.size)
      val resource = resourceList.map(_.toLowerCase)
      println(resource.size)

    }
  }

}
