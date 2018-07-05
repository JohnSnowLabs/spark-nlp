package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.nlp.annotators.{Tokenizer,Normalizer}
import com.johnsnowlabs.nlp._
import org.scalatest._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

import SparkAccessor.spark.implicits._
import com.johnsnowlabs.util.Benchmark


trait   SymmetricDeleteBehaviors { this: FlatSpec =>

  val spellChecker = new SymmetricDeleteApproach()
    .setCorpus(ExternalResource("src/test/resources/spell/sherlockholmes.txt",
      ReadAs.LINE_BY_LINE,
      Map("tokenPattern" -> "[a-zA-Z]+")))
    .fit(DataBuilder.basicDataBuild("dummy"))

  def testSuggestions(): Unit = {
    // The code above should be executed first
    spellChecker.getSuggestedCorrections("problex")
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
      val correction = spellChecker.check(misspell).getOrElse(misspell)
      assert(correction == wordAnswer.head._2)
    }
  }

  def testSeveralChecks(wordAnswer:  Seq[(String, String)]): Unit = {
    s"symspell checker " should s"successfully correct several misspells" in {
      wordAnswer.foreach( wa => {
        val misspell = wa._1
        val correction = spellChecker.check(misspell).getOrElse(misspell)
        assert(correction == wa._2)
      })
    }
  }

  def testAccuracyChecks(wordAnswer: Seq[(String, String)]): Unit = {
    s"spell checker" should s" correct words with at least a fair accuracy" in {
      val result = wordAnswer.count(wa =>
        spellChecker.check(wa._1).getOrElse(wa._1) == wa._2) / wordAnswer.length.toDouble
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
      Benchmark.time("without dict") { model.transform(data).show(20, false) }

    }
  }

  def testBigPipelineDict(): Unit = {
    s"a SymSpellChecker annotator using a big pipeline and a dictionary" should "successfully correct words" in {
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
        .setDictionary("src/test/resources/spell/words.txt")

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
      Benchmark.time("with dict") { model.transform(data).show(20, false) }

    }
  }

  def testIndividualWords(): Unit = {
    s"a SymSpellChecker annotator with pipeline of individual words" should
      "successfully correct words with good accuracy" in {

      val corpusData = Seq.empty[String].toDS
      val path = "src/test/resources/spell/misspelled_words.csv"
      val data = SparkAccessor.spark.read.format("csv").option("header", "true").load(path)

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

      Benchmark.time("without dictionary") { //to measure proceesing time
        var correctedData = model.transform(data)
        correctedData = correctedData.withColumn("prediction",
          when(col("word") === col("finished_spell"), 1).otherwise(0))
        val rightCorrections = correctedData.filter(col("prediction")===1).count()
        val wrongCorrections = correctedData.filter(col("prediction")===0).count()
        printf("Right Corrections: %d \n", rightCorrections)
        printf("Wrong Corrections: %d \n", wrongCorrections)
        val accuracy = rightCorrections.toFloat/(rightCorrections+wrongCorrections).toFloat
        printf("Accuracy: %f\n", accuracy)
      }
    }
  }

  def testIndividualWordsWithDictionary(): Unit = {
    s"a SymSpellChecker annotator with pipeline of individual words with dictionary" should
      "successfully correct words with good accuracy" in {

      val corpusData = Seq.empty[String].toDS
      val path = "src/test/resources/spell/misspelled_words.csv"
      val data = SparkAccessor.spark.read.format("csv").option("header", "true").load(path)

      val documentAssembler = new DocumentAssembler()
        .setInputCol("misspell")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val corpusPath = "src/test/resources/spell/sherlockholmes.txt"

      val spell = new SymmetricDeleteApproach()
        .setInputCols(Array("token"))
        .setOutputCol("spell")
        .setCorpus(ExternalResource(corpusPath,
          ReadAs.LINE_BY_LINE,
          Map("tokenPattern" -> "[a-zA-Z]+")))
        .setDictionary("src/test/resources/spell/words.txt")

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

      Benchmark.time("with dictionary") { //to measure processing time
        var correctedData = model.transform(data)
        correctedData = correctedData.withColumn("prediction",
          when(col("word") === col("finished_spell"), 1).otherwise(0))
        val rightCorrections = correctedData.filter(col("prediction")===1).count()
        val wrongCorrections = correctedData.filter(col("prediction")===0).count()
        printf("Right Corrections: %d \n", rightCorrections)
        printf("Wrong Corrections: %d \n", wrongCorrections)
        val accuracy = rightCorrections.toFloat/(rightCorrections+wrongCorrections).toFloat
        printf("Accuracy: %f\n", accuracy)
      }
    }
  }

  def testDatasetBasedSpellChecker(): Unit = {
    s"a SpellChecker annotator trained with datasets" should "successfully correct words" in {
      val corpusPath = "src/test/resources/spell/sherlockholmes.txt"
      val corpusData = SparkAccessor.spark.read.textFile(corpusPath)

      val dataPath = "src/test/resources/spell/misspelled_words.csv"
      val data = SparkAccessor.spark.read.format("csv").option("header", "true").load(dataPath)

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val normalizer = new Normalizer()
        .setInputCols(Array("token"))
        .setOutputCol("normal")
        .setLowercase(true)

      val spell = new SymmetricDeleteApproach()
        .setInputCols(Array("normal"))
        .setDictionary("src/test/resources/spell/words.txt")
        .setOutputCol("spell")

      val finisher = new Finisher()
        .setInputCols("spell")
        .setOutputAsArray(false)
        .setIncludeMetadata(false)

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          normalizer,
          spell,
          finisher
        ))

      /**Not cool to do this. Fit calls transform early, and will look for text column. Spark limitation...*/
      //corpusData.show()
      val model = pipeline.fit(corpusData.select(corpusData.col("value").as("text")))

      Benchmark.time("without dictionary") { //to measure processing time
        val df = data.select(data.col("misspell").as("text"),
                             data.col("word"))

        var correctedData = model.transform(df)
        correctedData = correctedData.withColumn("prediction",
          when(col("word") === col("finished_spell"), 1).otherwise(0))
        val rightCorrections = correctedData.filter(col("prediction")===1).count()
        val wrongCorrections = correctedData.filter(col("prediction")===0).count()
        printf("Right Corrections: %d \n", rightCorrections)
        printf("Wrong Corrections: %d \n", wrongCorrections)
        val accuracy = rightCorrections.toFloat/(rightCorrections+wrongCorrections).toFloat
        printf("Accuracy: %f\n", accuracy)
      } //End benchmark

    }
  }

  def testDatasetBasedSpellCheckerWithDic(): Unit = {
    s"a SpellChecker annotator trained with datasets and a dictionary" should "successfully correct words" in {
      val corpusPath = "src/test/resources/spell/sherlockholmes.txt"
      val corpusData = SparkAccessor.spark.read.textFile(corpusPath)

      val dataPath = "src/test/resources/spell/misspelled_words.csv"
      val data = SparkAccessor.spark.read.format("csv").option("header", "true").load(dataPath)

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val normalizer = new Normalizer()
        .setInputCols(Array("token"))
        .setOutputCol("normal")
        .setLowercase(true)

        val spell = new SymmetricDeleteApproach()
        .setInputCols(Array("normal"))
        .setDictionary("src/test/resources/spell/words.txt")
        .setOutputCol("spell")

      val finisher = new Finisher()
        .setInputCols("spell")
        .setOutputAsArray(false)
        .setIncludeMetadata(false)

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

      Benchmark.time("without dictionary") { //to measure processing time
        val df = data.select(data.col("misspell").as("text"),
          data.col("word"))

        var correctedData = model.transform(df)
        correctedData = correctedData.withColumn("prediction",
          when(col("word") === col("finished_spell"), 1).otherwise(0))
        val rightCorrections = correctedData.filter(col("prediction")===1).count()
        val wrongCorrections = correctedData.filter(col("prediction")===0).count()
        printf("Right Corrections: %d \n", rightCorrections)
        printf("Wrong Corrections: %d \n", wrongCorrections)
        val accuracy = rightCorrections.toFloat/(rightCorrections+wrongCorrections).toFloat
        printf("Accuracy: %f\n", accuracy)
      } // End benchmark

    }
  }

  def testLoadModel(): Unit = {
    s"a SymSpellChecker annotator with load model of" should
      "successfully correct words" in {
      val data = Seq("Hello World").toDS.toDF("text")

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val normalizer = new Normalizer()
        .setInputCols(Array("token"))
        .setOutputCol("normal")
        .setLowercase(true)

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          normalizer
        ))
      val pdata = pipeline.fit(Seq.empty[String].toDF("text")).transform(data)
      val spell = new SymmetricDeleteApproach()
        .setInputCols(Array("token"))
        .setOutputCol("spell")
        .setCorpus(ExternalResource("src/test/resources/spell/sherlockholmes.txt",
          ReadAs.LINE_BY_LINE,
          Map("tokenPattern" -> "[a-zA-Z]+")))
      val tspell = spell.fit(pdata)
      tspell.write.overwrite.save("./tmp_symspell")
      val modelSymSpell = SymmetricDeleteModel.load("./tmp_symspell")
      println("SymSpell Checker")
      modelSymSpell.transform(pdata).show(5)
    }
  }


  def testEmptyDataset(): Unit = {
    it should "rise an error message" in {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normal")
      .setLowercase(true)

    val spell = new SymmetricDeleteApproach()
      .setInputCols(Array("normal"))
      .setDictionary("src/test/resources/spell/words.txt")
      .setOutputCol("spell")

    val finisher = new Finisher()
      .setInputCols("spell")
      .setOutputAsArray(false)
      .setIncludeMetadata(false)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        normalizer,
        spell,
        finisher
      ))

      import SparkAccessor.spark.implicits._

      val emptyDataset = Seq.empty[String].toDF("text")

      val thrown = intercept[Exception] {
        pipeline.fit(emptyDataset)
      }
      val message = "corpus not provided and dataset for training is empty"
      assert(thrown.getMessage === "requirement failed: " + message)
    }
  }

}
