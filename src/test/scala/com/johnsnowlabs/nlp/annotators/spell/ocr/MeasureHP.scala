package com.johnsnowlabs.nlp.annotators.spell.ocr

import com.johnsnowlabs.nlp.annotators.spell.ocr.parser.{DateToken, NumberToken}
import com.johnsnowlabs.nlp.{Annotation, SparkAccessor}

object MeasureHP extends App with OcrTestData {
  override val correctFile = "/home/jose/auxdata/spell_dataset/correct_text.txt"
  override val rawFile = "/home/jose/auxdata/spell_dataset/raw_text.txt"

  val (correct, raw) = loadDataSets()

  val trainCorpusPath = "../auxdata/spell_dataset/vocab/bigone.txt"
  val langModelPath = "../auxdata/"

  import SparkAccessor.spark.implicits._
  val ocrspellModel = new OcrSpellCheckApproach().
    setWeights("distance.psv").
    setInputCols("text").
    setTrainCorpusPath(trainCorpusPath).
    setSpecialClasses(List(DateToken, NumberToken, AgeToken, UnitToken, MedicationClass)).
    fit(Seq.empty[String].toDF("text"))

  ocrspellModel.readModel(langModelPath, SparkAccessor.spark, "", useBundle=true)

  val time = System.nanoTime()
  val corrected = raw.map { tokens =>
    ocrspellModel.annotate(Seq(
      Annotation(tokens.mkString(" ")))).head.result.split(" ")
  }
  print(s"Done, ${(System.nanoTime() - time) / 1e9}")

  var destroyed = 0
  var totalErrorFree = 0
  var succeeded = 0
  var totalErroneous = 0

  (correct, raw, corrected).zipped.foreach { case (c, r, cd) =>
    (c, r, cd).zipped.foreach{ case (correctWord, rawWord, correctedWord) =>

      // the corrected word was ok at the output, and it was misspelled
      if (correctedWord.equals(correctWord) && !correctWord.equals(rawWord))
          succeeded += 1

      // we received a good word, destroyed it
      if (!correctedWord.equals(correctWord) && correctWord.equals(rawWord))
          destroyed += 1

      if (correctWord.equals(rawWord))
        totalErrorFree += 1

      if (!correctWord.equals(rawWord))
        totalErroneous += 1

      // words that we couldn't correct
      if(!correctedWord.equals(correctWord))
        println(rawWord, correctedWord, correctWord)
    }
  }

  println(s"accuracy % : ${(succeeded + totalErrorFree - destroyed).toDouble / (totalErroneous + totalErrorFree)}")
  println(s"total % of good corrections: ${succeeded.toFloat / totalErroneous}")
  println(s"total % of destructive corrections: ${destroyed.toFloat / totalErrorFree}")

}