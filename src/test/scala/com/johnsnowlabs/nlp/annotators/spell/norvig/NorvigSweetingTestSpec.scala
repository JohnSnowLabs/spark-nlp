package com.johnsnowlabs.nlp.annotators.spell.norvig

import com.johnsnowlabs.nlp.annotators.spell.util.Utilities
import com.johnsnowlabs.nlp.{AnnotatorBuilder, ContentProvider, DataBuilder}
import org.scalatest._

class NorvigSweetingTestSpec extends FlatSpec with NorvigSweetingBehaviors {

  System.gc()

  "A norvig sweeting approach" should behave like includeScoreOnMetadata

  "A norvig sweeting approach" should behave like testDefaultTokenCorpusParameter

  "An isolated spell checker" should behave like isolatedNorvigChecker(
    Seq(
      ("mral", "meal"),
      ("delicatly", "delicately"),
      ("efusive", "effusive"),
      ("lauging", "laughing"),
      ("juuuuuuuuuuuuuuuussssssssssttttttttttt", "just"),
      ("screeeeeeeewed", "screwed"),
      ("readampwritepeaceee", "readampwritepeaceee")
    )
  )

  "A spark spell checker" should behave like sparkBasedSpellChecker(DataBuilder.basicDataBuild("efusive", "destroyd"))

  "A spark spell checker using dataframes" should behave like sparkBasedSpellChecker(DataBuilder.basicDataBuild("efusive", "destroyd"), "TXTDS")

  "A good sized dataframe" should behave like sparkBasedSpellChecker(
    AnnotatorBuilder.withDocumentAssembler(ContentProvider.parquetData.limit(5000))
  )

  "A good sized dataframe trained with dataframe" should behave like datasetBasedSpellChecker

  "A norvig spell checker trained from fit" should behave like trainFromFitSpellChecker(Seq(
    "mral",
    "delicatly",
    "efusive",
    "lauging",
    "juuuuuuuuuuuuuuuussssssssssttttttttttt",
    "screeeeeeeewed",
    "readampwritepeaceee"
  ))

  it should behave like trainSpellCheckerModelFromFit

  it should behave like raiseErrorWhenWrongColumnIsSent

  "A norvig sweeting approach" should "get result by frequency" in {
    val wordsByFrequency = List(("eral", 1.toLong), ("mural" ,1.toLong), ("myal", 1.toLong),
                                ("kral", 2.toLong), ("oral", 2.toLong), ("maral", 2.toLong),
                                ("maral", 3.toLong), ("moral", 3.toLong), ("meal", 3.toLong))
    val maxValue = wordsByFrequency.maxBy(_._2)._2
    val bestWords = wordsByFrequency.filter(_._2 == maxValue).map(_._1)
    val expectedConfidenceValue = Utilities.computeConfidenceValue(bestWords)
    val norvigSweetingModel = new NorvigSweetingModel()

    val resultByFrequency = norvigSweetingModel.getResultByFrequency(wordsByFrequency)

    assert(bestWords.contains(resultByFrequency._1.getOrElse("")))
    assert(expectedConfidenceValue == resultByFrequency._2)

  }

  "A norvig sweeting approach" should "get result by hamming" in {
    val wordsByHamming = List(("eral", 1.toLong), ("myal" ,1.toLong), ("kral", 1.toLong),
                              ("oral", 1.toLong), ("meal", 1.toLong),
                              ("moral", 4.toLong), ("mural", 4.toLong), ("maral", 4.toLong))
    val minValue = wordsByHamming.minBy(_._2)._2
    val bestWords = wordsByHamming.filter(_._2 == minValue).map(_._1)
    val expectedConfidenceValue = Utilities.computeConfidenceValue(bestWords)
    val norvigSweetingModel = new NorvigSweetingModel()

    val resultByFrequency = norvigSweetingModel.getResultByHamming(wordsByHamming)

    assert(bestWords.contains(resultByFrequency._1.getOrElse("")))
    assert(expectedConfidenceValue == resultByFrequency._2)

  }

  "A norvig sweeting approach" should "get result by frequency and hamming" in {
    val input = "mral"
    val wordsByFrequency = List(("eral", 1.toLong), ("mural" ,1.toLong), ("myal", 1.toLong),
      ("kral", 2.toLong), ("maral", 2.toLong), ("maral", 2.toLong),
      ("oral", 3.toLong), ("moral", 3.toLong), ("meal", 3.toLong))
    val wordsByHamming = List(("eral", 1.toLong), ("myal" ,1.toLong), ("kral", 1.toLong),
      ("oral", 1.toLong), ("meal", 1.toLong),
      ("moral", 4.toLong), ("mural", 4.toLong), ("maral", 4.toLong))
    val norvigSweetingModel = new NorvigSweetingModel()
    val expectedResult = List ("oral", "meal")

    val resultByFrequencyAndHamming = norvigSweetingModel.getResult(wordsByFrequency, wordsByHamming, input)

    assert(expectedResult.contains(resultByFrequencyAndHamming._1))

  }

  "A norvig sweeting approach" should "get result by frequency and hamming with frequency priority" in {
    val input = "mral"
    val wordsByFrequency = List(("oral", 4.toLong), ("moral", 3.toLong), ("meal", 3.toLong))
    val wordsByHamming = List(("oral", 2.toLong), ("meal", 1.toLong),
                              ("moral", 4.toLong), ("mural", 4.toLong))
    val norvigSweetingModel = new NorvigSweetingModel().setFrequencyPriority(true)
    val expectedResult = "oral"

    val resultByFrequencyAndHamming = norvigSweetingModel.getResult(wordsByFrequency, wordsByHamming, input)

    assert(expectedResult.contains(resultByFrequencyAndHamming._1))

  }

  "A norvig sweeting approach" should "get result by frequency and hamming with hamming priority" in {
    val input = "mral"
    val wordsByFrequency = List(("oral", 4.toLong), ("moral", 3.toLong), ("meal", 3.toLong))
    val wordsByHamming = List(("oral", 2.toLong), ("meal", 1.toLong),
                              ("moral", 4.toLong), ("mural", 4.toLong))
    val norvigSweetingModel = new NorvigSweetingModel().setFrequencyPriority(false)
    val expectedResult = "meal"

    val resultByFrequencyAndHamming = norvigSweetingModel.getResult(wordsByFrequency, wordsByHamming, input)

    assert(expectedResult.contains(resultByFrequencyAndHamming._1))

  }

  "A norvig sweeting approach" should "get result by frequency or hamming" in {
    val input = "mral"
    val wordsByFrequency = List(("oral", 4.toLong))
    val wordsByHamming = List(("meal", 1.toLong))
    val norvigSweetingModel = new NorvigSweetingModel()
    val expectedResult = List("meal", "oral")

    val resultByFrequencyAndHamming = norvigSweetingModel.getResult(wordsByFrequency, wordsByHamming, input)

    assert(expectedResult.contains(resultByFrequencyAndHamming._1))

  }

}