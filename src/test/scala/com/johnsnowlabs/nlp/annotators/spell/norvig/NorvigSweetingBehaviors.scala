package com.johnsnowlabs.nlp.annotators.spell.norvig

import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

trait NorvigSweetingBehaviors { this: FlatSpec =>

  val spellChecker = new NorvigSweetingApproach()
    .setCorpus(ExternalResource("src/test/resources/spell/", ReadAs.LINE_BY_LINE, Map("tokenPattern" -> "[a-zA-Z]+")))
    .setDictionary("src/test/resources/spell/words.txt")
    //.setSlangDictionary(ExternalResource("src/test/resources/spell/slangs.txt", ReadAs.LINE_BY_LINE, Map("delimiter" -> ",")))
    .fit(DataBuilder.basicDataBuild("dummy"))

  def isolatedNorvigChecker(wordAnswer: Seq[(String, String)]): Unit = {
    s"spell checker" should s"correctly correct words" in {
      val result = wordAnswer.count(wa => spellChecker.check(wa._1) == wa._2) / wordAnswer.length.toDouble
      assert(result > 0.95, s"because result: $result did was below: 0.95")
    }
  }

  def sparkBasedSpellChecker(dataset: => Dataset[Row], inputFormat: String = "TXT"): Unit = {
    s"a SpellChecker Annotator with ${dataset.count} rows and corpus format $inputFormat" should s"successfully correct words" in {
      val result = AnnotatorBuilder.withFullSpellChecker(dataset, inputFormat)
        .select("document", "spell")
      result.show
      result.collect()
    }
  }

  def datasetBasedSpellChecker(): Unit = {
    s"a SpellChecker annotator trained with datasets" should "successfully correct words" in {
      val data = ContentProvider.parquetData.limit(1000)
      val corpusData = SparkAccessor.spark.read.textFile("src/test/resources/spell/sherlockholmes.txt")

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val normalizer = new Normalizer()
        .setInputCols(Array("token"))
        .setOutputCol("normal")

      val spell = new NorvigSweetingApproach()
        .setInputCols(Array("normal"))
        .setOutputCol("spell")
        .setDictionary("src/test/resources/spell/words.txt")
        //.setCorpus(ExternalResource("src/test/resources/spell/sherlockholmes.txt", ReadAs.LINE_BY_LINE, Map("tokenPattern" -> "[a-zA-Z]+")))

      val finisher = new Finisher()
        .setInputCols("spell")

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          normalizer,
          spell,
          finisher
        ))

      /**Not cool to do this. Fit calls transform early, and will look for text column. Spark limitation...*/
      val model = pipeline.fit(corpusData.select(corpusData.col("value").as("text")))
      model.transform(data).show()
    }
  }

  def testOutputSpellChecker(word: String): Unit = {
    s"spell checker" should s"correctly correct a word" in {
      val checkedWord = spellChecker.check(word)
      println(checkedWord)
    }
  }

}
