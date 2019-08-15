package com.johnsnowlabs.nlp.eval

import com.johnsnowlabs.nlp.annotator.NorvigSweetingApproach
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel
import com.johnsnowlabs.nlp.eval.spell.NorvigSpellEvaluation
import org.apache.spark.sql.SparkSession
import org.scalatest.FlatSpec

class NorvigSpellEvalTestSpec extends FlatSpec {

  private val spark = SparkSession.builder()
    .appName("benchmark")
    .master("local[*]")
    .config("spark.driver.memory", "8G")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  private val testFile = "./misspell.txt"
  private val groundTruthFile = "./ground_truth.txt"

  "A Norvig Spell Evaluation" should "display accuracy results" in {

    val trainFile = "src/test/resources/spell/sherlockholmes.txt"
    val dictionaryFile = "src/test/resources/spell/words.txt"

    val spell = new NorvigSweetingApproach()
      .setInputCols(Array("token"))
      .setOutputCol("checked")
      .setDictionary(dictionaryFile)

    val norvigSpellEvaluation = new NorvigSpellEvaluation(spark, testFile, groundTruthFile)
    norvigSpellEvaluation.computeAccuracyAnnotator(trainFile, spell)

  }

  "A Norvig Spell Evaluation" should "display accuracy results for pre-trained model" in {

    val spell = NorvigSweetingModel.pretrained()

    val norvigSpellEvaluation = new NorvigSpellEvaluation(spark, testFile, groundTruthFile)
    norvigSpellEvaluation.computeAccuracyModel(spell)

  }

}