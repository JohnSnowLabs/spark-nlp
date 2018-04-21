package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp._

import org.scalatest._
import org.apache.spark.ml.Pipeline

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

  def datasetBasedSSympellChecker(): Unit = {
    s"a SymSpellChecker annotator trained with datasets" should "successfully correct words" in {
      val data = ContentProvider.parquetData.limit(1000)
      //val corpusPath = "src/test/resources/spell/sherlockholmes.txt"
      //val corpusData = SparkAccessor.spark.read.textFile(corpusPath)

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val spell = new SymmetricDeleteApproach()
        .setInputCols(Array("normal"))
        .setOutputCol("spell")
        .setCorpus(ExternalResource("src/test/resources/spell/sherlockholmes.txt",
                                    ReadAs.LINE_BY_LINE, Map("tokenPattern" -> "[a-zA-Z]+")))

      val finisher = new Finisher()
        .setInputCols("spell")

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          spell,
          finisher
        ))

      val model = pipeline.fit(Seq.empty[String].toDS)
      model.transform(data).show()

    }
  }

}
