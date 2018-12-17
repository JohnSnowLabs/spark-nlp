package com.johnsnowlabs.example

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.{Tokenizer}
import com.johnsnowlabs.nlp.util.io.OcrHelper
import com.johnsnowlabs.nlp.SparkAccessor.spark

object SampleImageOCR extends App with Spark{

  import spark.implicits._

  val data = OcrHelper.createDataset(spark,"./data/hps/","region","metadata")

  val documentAssembler = new DocumentAssembler().
    setTrimAndClearNewLines(false).
    setInputCol("region").
    setOutputCol("document").
    setMetadataCol("metadata")

  val tokenizer: Tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  val spellChecker = ContextSpellCheckerModel.read
    .load("models/med_spell_checker")
    .setInputCols("token")
    .setOutputCol("checked")
    .setUseNewLines(true)

}
