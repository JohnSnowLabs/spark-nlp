package com.johnsnowlabs.nlp.annotators.spell.norvig

import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._
import com.johnsnowlabs.nlp.SparkAccessor.spark.implicits._
import com.johnsnowlabs.util.PipelineModels

trait NorvigSweetingBehaviors { this: FlatSpec =>

  private val trainDataSet = AnnotatorBuilder.getTrainingDataSet("src/test/resources/spell/sherlockholmes.txt")
  private val predictionDataSet = ContentProvider.parquetData.limit(1000)

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  private val normalizer = new Normalizer()
    .setInputCols(Array("token"))
    .setOutputCol("normal")

  private val spell = new NorvigSweetingApproach()
    .setInputCols(Array("token"))
    .setOutputCol("spell")
    .setDictionary("src/test/resources/spell/words.txt")

  private val finisher = new Finisher()
    .setInputCols("spell")

  private val pipeline = new Pipeline()
    .setStages(Array(
      documentAssembler,
      tokenizer,
      normalizer,
      spell,
      finisher
    ))


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

  def isolatedNorvigChecker(wordAnswer: Seq[(String, String)]): Unit = {
    s"spell checker" should s"correctly correct words" in {

      val trainBigDataSet = getTrainDataSet("src/test/resources/spell/")
      val spellChecker = new NorvigSweetingApproach()
        .setInputCols("token")
        .setOutputCol("spell")
        .setDictionary("src/test/resources/spell/words.txt")
        .fit(trainBigDataSet)

      val result = wordAnswer.count(wa => spellChecker.checkSpellWord(wa._1)._1 == wa._2) / wordAnswer.length.toDouble
      assert(result > 0.95, s"because result: $result did was below: 0.95")
    }
  }

  def sparkBasedSpellChecker(dataset: => Dataset[Row], inputFormat: String = "TXT"): Unit = {
    s"a SpellChecker Annotator with ${dataset.count} rows and corpus format $inputFormat" should
      s"successfully correct words" in {
      val result = AnnotatorBuilder.withFullSpellChecker(dataset, inputFormat)
        .select("document", "spell")
      result.collect()
    }
  }

  def datasetBasedSpellChecker(): Unit = {
    s"a SpellChecker annotator trained with datasets" should "successfully correct words" in {

      /**Not cool to do this. Fit calls transform early, and will look for text column. Spark limitation...*/
      val model = pipeline.fit(trainDataSet)
      model.transform(predictionDataSet).show()
    }
  }

  def testDefaultTokenCorpusParameter(): Unit = {
    s"using a corpus with default token parameter" should "successfully correct words" in {

      /**Not cool to do this. Fit calls transform early, and will look for text column. Spark limitation...*/
      val model = pipeline.fit(trainDataSet)
      model.transform(predictionDataSet).show()
    }
  }

  def trainFromFitSpellChecker(words: Seq[String]): Unit = {
    val trainBigDataSet = getTrainDataSet("src/test/resources/spell/")
    val predictionDataSet = words.toDF("text")

    val model = pipeline.fit(trainBigDataSet)
    model.transform(predictionDataSet).show()
  }

  def trainSpellCheckerModelFromFit(): Unit = {
    "" should "trained a model from fit when dataset with array annotation is used"  in {
      val trainDataSet = getTrainDataSet("src/test/resources/spell/sherlockholmes.txt")

      val spell = new NorvigSweetingApproach()
        .setInputCols(Array("token"))
        .setOutputCol("spell")
        .setDictionary("src/test/resources/spell/words.txt")
        .fit(trainDataSet)

      assert(spell.isInstanceOf[NorvigSweetingModel])

    }

  }

  def raiseErrorWhenWrongColumnIsSent(): Unit = {
    "" should "raise an error when dataset without array annotation is used"  in {
      val trainDataSet = AnnotatorBuilder.getTrainingDataSet("src/test/resources/spell/sherlockholmes.txt")
      val expectedErrorMessage = "Train dataset must have an array annotation type column"
      val spell = new NorvigSweetingApproach()
        .setInputCols(Array("text"))
        .setOutputCol("spell")
        .setDictionary("src/test/resources/spell/words.txt")

      val caught = intercept[IllegalArgumentException] {
        spell.fit(trainDataSet)
      }

      assert(caught.getMessage == expectedErrorMessage)
    }
  }

  def includeScoreOnMetadata(): Unit = {
    "" should "include score on metadata" in {
      val pipeline = new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          normalizer,
          spell
        ))
      val model = pipeline.fit(trainDataSet)
      model.transform(predictionDataSet).select("spell").show(false)
    }
  }

}
