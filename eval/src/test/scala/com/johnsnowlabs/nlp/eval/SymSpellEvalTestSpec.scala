package com.johnsnowlabs.nlp.eval

import com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach
import org.scalatest.FlatSpec

class SymSpellEvalTestSpec extends FlatSpec {

  "A Symnmetric Spell Evaluation" should "display accuracy results" in {

    val trainFile = "src/test/resources/spell/sherlockholmes.txt"
    val dictionaryFile = "src/test/resources/spell/words.txt"
    val testFile = "./misspell.txt"
    val groundTruthFile = "./ground_truth.txt"

    val spell = new SymmetricDeleteApproach()
      .setInputCols(Array("token"))
      .setOutputCol("spell")
      .setDictionary(dictionaryFile)

    SymSpellEvaluation(trainFile, spell, testFile, groundTruthFile)
  }

}
