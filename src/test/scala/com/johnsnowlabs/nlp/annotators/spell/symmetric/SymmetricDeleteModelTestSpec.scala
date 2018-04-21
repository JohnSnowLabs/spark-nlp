package com.johnsnowlabs.nlp.annotators.spell.symmetric


import org.scalatest._

class SymmetricDeleteModelTestSpec extends FlatSpec with SymmetricDeleteBehaviors {


  //testSuggestions()

  //testLevenshteinDistance()


  "a simple isolated symmetric spell checker " should behave like testSimpleCheck(
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


  /* With this got 0.20, we should check it using another corpus
  ("contenpted", "contented"),
    ("dirven", "driven"),
    ("biscuts", "biscuits"),
    ("remeber", "remember"),
    ("inconvienient", "inconvenient")
  * */

  "a good sized dataframe trained with Spark pipeline" should behave like datasetBasedSSympellChecker

}
