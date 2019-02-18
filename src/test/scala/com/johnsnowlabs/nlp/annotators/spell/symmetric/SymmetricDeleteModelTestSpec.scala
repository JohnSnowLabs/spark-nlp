package com.johnsnowlabs.nlp.annotators.spell.symmetric


import org.scalatest._

class SymmetricDeleteModelTestSpec extends FlatSpec with SymmetricDeleteBehaviors {


  //testSuggestions()

  //testLevenshteinDistance()

  "A symmetric delete approach" should behave like testDefaultTokenCorpusParameter

  "a lower case word " should behave like obtainLowerCaseTypeFromALowerCaseWord

  it should behave like transformLowerCaseTypeWordIntoLowerCase

  "a Capital case word " should behave like obtainCapitalCaseTypeFromACapitalCaseWord

  it should behave like transformUpperCaseTypeWordIntoUpperCase

  "an UPPER case word " should behave like obtainUpperCaseTypeFromUpperCaseWord

  it should behave like transformUpperCaseTypeWordIntoUpperCase

  "a noise word " should behave like returnTrueWhenWordIsNoisy

  "a not noise word" should behave like returnFalseWhenWordIsNotNoisy

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
    ("gr8", "great"),
    ("juuuuuuuuuuuuuuuussssssssssttttttttttt", "just"),
    ("screeeeeeeewed", "screwed"),
    ("readampwritepeaceee", "readampwritepeaceee")
  ))

  "a good sized dataframe with Spark pipeline" should behave like testBigPipeline

  "a good sized dataframe with Spark pipeline and spell checker dictionary" should behave like testBigPipelineDict

  "a dataset of individual words with Spark pipeline" should behave like testIndividualWords

  "a dataset of individual words with Spark pipeline using a dictionary" should behave like
    testIndividualWordsWithDictionary

  "a simple isolated symmetric spell checker with dataset" should behave like testDatasetBasedSpellChecker

  "a simple isolated symmetric spell checker with dataset using dictionary" should behave like
    testDatasetBasedSpellCheckerWithDic

  "a loaded model " should behave like testLoadModel()

  "a symmetric spell checker with empty dataset" should behave like testEmptyDataset

}
