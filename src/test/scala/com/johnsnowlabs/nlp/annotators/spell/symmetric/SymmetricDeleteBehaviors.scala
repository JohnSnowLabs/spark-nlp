package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.nlp._
import org.scalatest._


trait SymmetricDeleteBehaviors { this: FlatSpec =>

  val spellChecker = new SymmetricDeleteApproach()
      .setCorpus(ExternalResource("src/test/resources/spell/sherlockholmes.txt",
                 ReadAs.LINE_BY_LINE,
                 Map("tokenPattern" -> "[a-zA-Z]+")))
      .fit(DataBuilder.basicDataBuild("dummy"))

  def testSuggestions(): Unit = {
    // The code above should be executed first
    val silent = true
    spellChecker.getSuggestedCorrections("problex", silent)
    println("done")
  }

  def testLevenshteinDistance(): Unit = {
    printDistance("kitten", "sitting")
    printDistance("rosettacode", "raisethysword")
  }

  private def printDistance(s1:String, s2:String): Unit =
    println("%s -> %s : %d".format(s1, s2, spellChecker.levenshteinDistance(s1, s2)))

  def testCheck(): Unit ={
    val misspell = List("contende", "problex", "pronounciation", "localy", "compair")
    misspell.foreach(ms =>{
      val word = spellChecker.check(ms)
      println(word)
    })
    println("Test passed")

    /*val misspell = "Problex"
    val word = spellChecker.check(misspell)
    println(word)*/
  }

}
