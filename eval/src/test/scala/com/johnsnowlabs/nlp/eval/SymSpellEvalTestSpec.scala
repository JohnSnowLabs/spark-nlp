package com.johnsnowlabs.nlp.eval

import com.johnsnowlabs.nlp.annotators.spell.symmetric.{SymmetricDeleteApproach, SymmetricDeleteModel}
import org.scalatest.FlatSpec

class SymSpellEvalTestSpec extends FlatSpec {

  "A Symnmetric Spell Evaluation" should "display accuracy results" in {

    val trainFile = "src/test/resources/spell/sherlockholmes.txt"
    val dictionaryFile = "src/test/resources/spell/words.txt"
    val groundTruthFile = "./ground_truth.txt"
    val testFile = "./misspell.txt"

    val spell = new SymmetricDeleteApproach()
      .setInputCols(Array("token"))
      .setOutputCol("checked")
      .setDictionary(dictionaryFile)

    val symSpellEvaluation = new SymSpellEvaluation(testFile, groundTruthFile)
    symSpellEvaluation.computeAccuracyAnnotator(trainFile, spell)
  }

  "A Symnmetric Spell Evaluation" should "display accuracy results for a pre-trained model" in {

    val groundTruthFile = "./ground_truth.txt"
    val testFile = "./misspell.txt"

    val spell = SymmetricDeleteModel.pretrained()

    val symSpellEvaluation = new SymSpellEvaluation(testFile, groundTruthFile)
    symSpellEvaluation.computeAccuracyModel(spell)
  }

}