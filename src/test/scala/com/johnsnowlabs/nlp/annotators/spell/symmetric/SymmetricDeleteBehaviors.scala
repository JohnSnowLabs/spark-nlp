package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp._
import org.scalatest._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions._
import SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp.annotators.spell.common.LevenshteinDistance
import com.johnsnowlabs.util.{Benchmark, PipelineModels}
import org.apache.spark.sql.{DataFrame, Dataset}


trait SymmetricDeleteBehaviors extends LevenshteinDistance { this: FlatSpec =>

  private val trainDataSet = AnnotatorBuilder.getTrainingDataSet("src/test/resources/spell/sherlockholmes.txt")
  private val predictionDataSet = ContentProvider.parquetData.limit(500)

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  private val spell = new SymmetricDeleteApproach()
    .setInputCols(Array("token"))
    .setOutputCol("spell")
    .setDictionary("src/test/resources/spell/words.txt")

  private val finisher = new Finisher()
    .setInputCols("spell")

  private val pipeline = new Pipeline()
    .setStages(Array(
      documentAssembler,
      tokenizer,
      spell,
      finisher
    ))

  private val CAPITAL = 'C'
  private val LOWERCASE = 'L'
  private val UPPERCASE = 'U'

  def getTrainDataSet(trainDataPath: String): Dataset[_] = {

    val emptyDataSet = PipelineModels.dummyDataset
    val corpusDataSet = AnnotatorBuilder.getTrainingDataSet(trainDataPath)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer
      ))

    pipeline.fit(emptyDataSet).transform(corpusDataSet)
  }

  def testSimpleCheck(wordAnswer: Seq[(String, String)]): Unit ={
     it should "successfully correct a misspell" in {
       val trainDataSet = getTrainDataSet("src/test/resources/spell/sherlockholmes.txt")
       val spellChecker = new SymmetricDeleteApproach()
         .setInputCols("token")
         .setOutputCol("spell")
         .fit(trainDataSet)
       val misspell = wordAnswer.head._1

      val correction = spellChecker.check(misspell).getOrElse(misspell)
      assert(correction == wordAnswer.head._2)
    }
  }

  def testSeveralChecks(wordAnswer:  Seq[(String, String)]): Unit = {
    s"symspell checker " should s"successfully correct several misspells" in {

      val trainDataSet = getTrainDataSet("src/test/resources/spell/sherlockholmes.txt")
      val spellChecker = new SymmetricDeleteApproach()
        .setInputCols("token")
        .setOutputCol("spell")
        .fit(trainDataSet)

      wordAnswer.foreach( wa => {
        val misspell = wa._1
        val correction = spellChecker.check(misspell).getOrElse(misspell)
        assert(correction == wa._2)
      })
    }
  }

  def testAccuracyChecks(wordAnswer: Seq[(String, String)]): Unit = {
    s"spell checker" should s" correct words with at least a fair accuracy" in {
      val trainDataSet = getTrainDataSet("src/test/resources/spell/sherlockholmes.txt")
      val spellChecker = new SymmetricDeleteApproach()
        .setInputCols("token")
        .setOutputCol("spell")
        .fit(trainDataSet)

      val result = wordAnswer.count(wa =>
        spellChecker.check(wa._1).getOrElse(wa._1) == wa._2) / wordAnswer.length.toDouble
      println(result)
      assert(result > 0.60, s"because result: $result did was below: 0.60")
    }
  }


  def testBigPipeline(): Unit = {
    s"a SymSpellChecker without dictionary" should "successfully correct words" in {

      val spell = new SymmetricDeleteApproach()
        .setInputCols(Array("token"))
        .setOutputCol("spell")

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          spell,
          finisher
        ))

      val model = pipeline.fit(trainDataSet)
      
      assert(model.transform(predictionDataSet).isInstanceOf[DataFrame])
    }
  }

  def testBigPipelineDict(): Unit = {
    s"a SymSpellChecker annotator with a dictionary" should "successfully correct words" in {

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          spell,
          finisher
        ))

      val model = pipeline.fit(trainDataSet)

      assert(model.transform(predictionDataSet).isInstanceOf[DataFrame])

    }
  }

//  def testIndividualWords(): Unit = {
//    s"a SymSpellChecker annotator with pipeline of individual words" should
//      "successfully correct words with good accuracy" in {
//
//      val path = "src/test/resources/spell/misspelled_words.csv"
//      val predictionDataSet = SparkAccessor.spark.read.format("csv").option("header", "true").load(path)
//
//      val documentAssembler = new DocumentAssembler()
//        .setInputCol("misspell")
//        .setOutputCol("document")
//
//      val spell = new SymmetricDeleteApproach()
//        .setInputCols(Array("token"))
//        .setOutputCol("spell")
//
//      val finisher = new Finisher()
//        .setInputCols("spell")
//        .setOutputAsArray(false)
//        .setAnnotationSplitSymbol("@")
//        .setValueSplitSymbol("#")
//
//      val pipeline = new Pipeline()
//        .setStages(Array(
//          documentAssembler,
//          tokenizer,
//          spell,
//          finisher
//        ))
//
//      val model = pipeline.fit(trainDataSet.select(trainDataSet.col("text").as("misspell")))
//
//      Benchmark.time("without dictionary") { //to measure proceesing time
//        var correctedData = model.transform(predictionDataSet)
//        correctedData = correctedData.withColumn("prediction",
//          when(col("word") === col("finished_spell"), 1).otherwise(0))
//        val rightCorrections = correctedData.filter(col("prediction")===1).count()
//        val wrongCorrections = correctedData.filter(col("prediction")===0).count()
//        printf("Right Corrections: %d \n", rightCorrections)
//        printf("Wrong Corrections: %d \n", wrongCorrections)
//        val accuracy = rightCorrections.toFloat/(rightCorrections+wrongCorrections).toFloat
//        printf("Accuracy: %f\n", accuracy)
//      }
//    }
//  }

  def testIndividualWordsWithDictionary(): Unit = {
    s"a SymSpellChecker annotator with pipeline of individual words with dictionary" should
      "successfully correct words with good accuracy" in {

      val path = "src/test/resources/spell/misspelled_words.csv"
      val predictionDataSet = SparkAccessor.spark.read.format("csv").option("header", "true").load(path)

//      val documentAssembler = new DocumentAssembler()
//        .setInputCol("misspell")
//        .setOutputCol("document")
//
//      val tokenizer = new Tokenizer()
//        .setInputCols(Array("document"))
//        .setOutputCol("token")
//
//      val spell = new SymmetricDeleteApproach()
//        .setInputCols(Array("token"))
//        .setOutputCol("spell")
//        .setDictionary("src/test/resources/spell/words.txt")
//
//      val finisher = new Finisher()
//        .setInputCols("spell")
//        .setOutputAsArray(false)
//        .setAnnotationSplitSymbol("@")
//        .setValueSplitSymbol("#")

      val model = pipeline.fit(trainDataSet.select(trainDataSet.col("text").as("misspell")))

      Benchmark.time("with dictionary") { //to measure processing time
        var correctedData = model.transform(predictionDataSet)
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

  def testLoadModel(): Unit = {
    s"a SymSpellChecker annotator with load model of" should
      "successfully correct words" in {
      val predictionDataSet = Seq("Hello World").toDS.toDF("text")

      val spell = new SymmetricDeleteApproach()
        .setInputCols(Array("token"))
        .setOutputCol("spell")

      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          spell
        ))

      val model = pipeline.fit(trainDataSet)
      model.write.overwrite.save("./tmp_symspell")
      val modelSymSpell = PipelineModel.load("./tmp_symspell")

      assert(modelSymSpell.transform(predictionDataSet).isInstanceOf[DataFrame])
    }
  }

  def testEmptyDataset(): Unit = {
    it should "rise an error message" in {

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

  def transformCapitalCaseTypeWordIntoCapitalCase(): Unit = {
    it should "transform a corrected capital case type word into capital case" in {
      val trainDataSet = getTrainDataSet("src/test/resources/spell/sherlockholmes.txt")
      val spellChecker = new SymmetricDeleteApproach()
        .setInputCols("token")
        .setOutputCol("spell")
        .fit(trainDataSet)
      val caseType = CAPITAL
      val correctedWord = "problem"
      val expectedWord = "Problem"

      val transformedWord = spellChecker.transformToOriginalCaseType(caseType, correctedWord)

      assert(transformedWord === expectedWord)

    }
  }

  def testDefaultTokenCorpusParameter(): Unit = {
    s"using a corpus with default token parameter" should "successfully correct words" in {

      val spell = new SymmetricDeleteApproach()
        .setInputCols(Array("token"))
        .setOutputCol("spell")
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

      val model = pipeline.fit(trainDataSet)

      assert(model.transform(predictionDataSet).isInstanceOf[DataFrame])

    }
  }

  def trainFromFitSpellChecker(words: Seq[String]): Unit = {

    val predictionDataSet = words.toDF("text")

    val model = pipeline.fit(trainDataSet)
    model.transform(predictionDataSet).show()
  }

  def trainSpellCheckerModelFromFit(): Unit = {
    "" should "trained a model from fit when dataset with array annotation is used"  in {
      val trainDataSet = getTrainDataSet("src/test/resources/spell/sherlockholmes.txt")

      val spell = new SymmetricDeleteApproach()
        .setInputCols(Array("token"))
        .setOutputCol("spell")
        .setDictionary("src/test/resources/spell/words.txt")
        .fit(trainDataSet)

      assert(spell.isInstanceOf[SymmetricDeleteModel])

    }

  }

  def raiseErrorWhenWrongColumnIsSent(): Unit = {
    "" should "raise an error when dataset without array annotation is used"  in {
      val trainDataSet = AnnotatorBuilder.getTrainingDataSet("src/test/resources/spell/sherlockholmes.txt")
      val expectedErrorMessage = "Train dataset must have an array annotation type column"
      val spell = new SymmetricDeleteApproach()
        .setInputCols(Array("text"))
        .setOutputCol("spell")
        .setDictionary("src/test/resources/spell/words.txt")

      val caught = intercept[IllegalArgumentException] {
        spell.fit(trainDataSet)
      }

      assert(caught.getMessage == expectedErrorMessage)
    }
  }

}
