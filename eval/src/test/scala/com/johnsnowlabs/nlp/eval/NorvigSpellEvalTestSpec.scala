package com.johnsnowlabs.nlp.eval

import com.johnsnowlabs.nlp.annotator.NorvigSweetingApproach
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel
import com.johnsnowlabs.nlp.eval.spell.NorvigSpellEvaluation
import org.scalatest.FlatSpec

class NorvigSpellEvalTestSpec extends FlatSpec {

  "A Norvig Spell Evaluation" should "display accuracy results" in {

    val trainFile = "src/test/resources/spell/sherlockholmes.txt"
    val dictionaryFile = "src/test/resources/spell/words.txt"
    val groundTruthFile = "./ground_truth.txt"
    val testFile = "./misspell.txt"

    val spell = new NorvigSweetingApproach()
      .setInputCols(Array("token"))
      .setOutputCol("checked")
      .setDictionary(dictionaryFile)

    val norvigSpellEvaluation = new NorvigSpellEvaluation(testFile, groundTruthFile)
    norvigSpellEvaluation.computeAccuracyAnnotator(trainFile, spell)

  }

  "A Norvig Spell Evaluation" should "display accuracy results for pre-trained model" in {

    val groundTruthFile = "./ground_truth.txt"
    val testFile = "./misspell.txt"

    val spell = NorvigSweetingModel.pretrained()

    val norvigSpellEvaluation = new NorvigSpellEvaluation(testFile, groundTruthFile)
    norvigSpellEvaluation.computeAccuracyModel(spell)

  }

}