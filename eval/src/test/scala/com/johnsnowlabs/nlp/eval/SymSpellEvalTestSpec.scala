package com.johnsnowlabs.nlp.eval

import com.johnsnowlabs.nlp.annotators.spell.symmetric.{SymmetricDeleteApproach, SymmetricDeleteModel}
import com.johnsnowlabs.nlp.eval.spell.SymSpellEvaluation
import org.apache.spark.sql.SparkSession
import org.scalatest.FlatSpec

class SymSpellEvalTestSpec extends FlatSpec {

  private val spark = SparkSession.builder()
    .appName("benchmark")
    .master("local[*]")
    .config("spark.driver.memory", "8G")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  private val testFile = "./misspell.txt"
  private val groundTruthFile = "./ground_truth.txt"

  "A Symnmetric Spell Evaluation" should "display accuracy results" in {

    val trainFile = "src/test/resources/spell/sherlockholmes.txt"
    val dictionaryFile = "src/test/resources/spell/words.txt"

    val spell = new SymmetricDeleteApproach()
      .setInputCols(Array("token"))
      .setOutputCol("checked")
      .setDictionary(dictionaryFile)

    val symSpellEvaluation = new SymSpellEvaluation(spark, testFile, groundTruthFile)
    symSpellEvaluation.computeAccuracyAnnotator(trainFile, spell)
  }

  "A Symnmetric Spell Evaluation" should "display accuracy results for a pre-trained model" in {

    val spell = SymmetricDeleteModel.pretrained()

    val symSpellEvaluation = new SymSpellEvaluation(spark, testFile, groundTruthFile)
    symSpellEvaluation.computeAccuracyModel(spell)
  }

}