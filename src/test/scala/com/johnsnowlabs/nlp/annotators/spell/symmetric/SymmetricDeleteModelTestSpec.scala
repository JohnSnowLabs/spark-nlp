package com.johnsnowlabs.nlp.annotators.spell.symmetric


import org.scalatest._

class SymmetricDeleteModelTestSpec(var foo: String = "pruebas") extends FlatSpec with SymmetricDeleteBehaviors {


  //testSuggestions()

  //testLevenshteinDistance()


  /*"a simple isolated symmetric spell checker " should behave like testSimpleCheck(
    Seq(("problex", "problem")))

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
    testIndividualWordsWithDictionary*/

  "a simple isolated symmetric spell checker with dataset" should behave like datasetBasedSpellChecker

  // "a spark dataset " should behave like testSparkDataset()

}
