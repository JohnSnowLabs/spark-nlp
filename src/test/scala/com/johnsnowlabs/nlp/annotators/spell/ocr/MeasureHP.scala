package com.johnsnowlabs.nlp.annotators.spell.ocr

import com.johnsnowlabs.nlp.SparkAccessor.spark
import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp.annotators.spell.ocr.parser.{DateToken, NumberToken}
import com.johnsnowlabs.nlp.util.io.OcrHelper
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, SparkAccessor}
import org.apache.spark.ml.Pipeline
import SparkAccessor.spark.implicits._

object MeasureHP extends App with OcrTestData {
  override val correctFile = "/home/jose/auxdata/spell_dataset/correct_text.txt"
  override val rawFile = "/home/jose/auxdata/spell_dataset/raw_text.txt"

  val (correct, raw) = loadDataSets()

  val trainCorpusPath = "../auxdata/spell_dataset/vocab/bigone.txt"
  val langModelPath = "../auxdata/"

  val data = OcrHelper.createDataset(spark,
    "ocr/src/test/resources/pdfs/h_and_p.pdf",
    "region", "metadata")

  val documentAssembler =
    new DocumentAssembler().
      setInputCol("region").
      setMetadataCol("metadata").
      setOutputCol("text")

  val tokenizer: Tokenizer = new Tokenizer()
    .setInputCols(Array("text"))
    .setOutputCol("token")

  val normalizer: Normalizer = new Normalizer()
    .setInputCols(Array("token"))
    .setOutputCol("normalized")

  val ocrspellModel = new OcrSpellCheckApproach().
    setWeights("distance.psv").
    setInputCols("normalized").
    setTrainCorpusPath(trainCorpusPath).
    setSpecialClasses(List(DateToken, NumberToken, AgeToken, UnitToken, MedicationClass)).
    fit(Seq.empty[String].toDF("text"))

  ocrspellModel.readModel(langModelPath, SparkAccessor.spark, "", useBundle=true)

  val pipeline = new Pipeline().
    setStages(Array(documentAssembler, tokenizer, normalizer, ocrspellModel)).
    fit(Seq.empty[String].toDF("region"))

  val time = System.nanoTime()
  val corrected = pipeline.transform(data).as[Seq[Annotation]].map(_.head.result).collect()
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