package com.johnsnowlabs.nlp.annotators.spell.symmetric


import org.scalatest._

class SymmetricDeleteModelTestSpec extends FlatSpec with SymmetricDeleteBehaviors {

  "A symmetric delete approach" should behave like testDefaultTokenCorpusParameter

  "a simple isolated symmetric spell checker with a lowercase word" should behave like testSimpleCheck(
    Seq(("problex", "problem")))

  "a simple isolated symmetric spell checker with a Capitalized word" should behave like testSimpleCheck(
    Seq(("Problex", "Problem")))

  "a simple isolated symmetric spell checker with UPPERCASE word" should behave like testSimpleCheck(
    Seq(("PROBLEX", "PROBLEM")))

  "a simple isolated symmetric spell checker with noisy word" should behave like testSimpleCheck(
    Seq(("CARDIOVASCULAR:", "CARDIOVASCULAR:")))

  "an isolated symmetric spell checker " should behave like testSeveralChecks(Seq(
    ("problex", "problem"),
    ("contende", "continue"),
    ("pronounciation", "pronounciation"),
    ("localy", "local"),
    ("compair", "company")
  ))

  "a symmetric spell checker " should behave like testAccuracyChecks(Seq(
    ("mral", "meal"),
    ("delicatly", "delicately"),
    ("efusive", "effusive"),
    ("lauging", "laughing"),
    ("juuuuuuuuuuuuuuuussssssssssttttttttttt", "just"),
    ("screeeeeeeewed", "screwed"),
    ("readampwritepeaceee", "readampwritepeaceee")
  ))

  "a good sized dataframe with Spark pipeline" should behave like testBigPipeline

  "a good sized dataframe with Spark pipeline and spell checker dictionary" should behave like testBigPipelineDict

  "a loaded model " should behave like testLoadModel()

  "a symmetric spell checker with empty dataset" should behave like testEmptyDataset

  "A symmetric spell checker trained from fit" should behave like trainFromFitSpellChecker(Seq(
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

}
