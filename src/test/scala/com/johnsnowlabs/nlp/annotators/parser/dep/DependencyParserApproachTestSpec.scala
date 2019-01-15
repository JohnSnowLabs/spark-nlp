package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition.{Sentence, WordData}
import org.scalatest.FlatSpec

class DependencyParserApproachTestSpec extends FlatSpec{


  "A Dependency Parser Approach" should "read CoNLL data from TreeBank format file" in {

    val filesContent = "Pierre\tNNP\t0\n"+
                        "is\tVBZ\t1\n"+
                        "old\tJJ\t2\n\n"+
                        "Vinken\tNNP\t0\n"+
                        "is\tVBZ\t1\n"+
                        "chairman\tNN\t2"
    val dependencyParserApproach = new DependencyParserApproach()
    val expectedResult: List[Sentence] = List(
      List(WordData("Pierre","NNP",3), WordData("is","VBZ",0), WordData("old","JJ",1)),
      List(WordData("Vinken","NNP",3), WordData("is","VBZ",0), WordData("chairman","NN",1))
    )

    val result = dependencyParserApproach.readCONLL(filesContent)

    assert(expectedResult == result)
  }

  //TODO: Add a Unit Test that load CoNLL UD, it seems we will require a new method.



}
