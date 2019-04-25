package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition.{Sentence, WordData}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.util.PipelineModels
import org.scalatest.FlatSpec

class DependencyParserApproachTestSpec extends FlatSpec{

  private val emptyDataSet = PipelineModels.dummyDataset

  "A dependency parser that sets TreeBank and CoNLL-U format files " should "raise an error" in {

    val dependencyParserApproach = new DependencyParserApproach()
      .setInputCols(Array("sentence", "pos", "token"))
      .setOutputCol("dependency")
      .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")
      .setConllU("src/test/resources/parser/unlabeled/conll-u")
      .setNumberOfIterations(10)
    val expectedErrorMessage = "Use either TreeBank or CoNLL-U format file both are not allowed."

    val caught = intercept[IllegalArgumentException]{
      dependencyParserApproach.fit(emptyDataSet)
    }

    assert(caught.getMessage == expectedErrorMessage)
  }

  "A dependency parser that does not set TreeBank or CoNLL-U format files " should "raise an error" in {

    val pipeline = new DependencyParserApproach()
    val expectedErrorMessage = "Either TreeBank or CoNLL-U format file is required."

    val caught = intercept[IllegalArgumentException]{
      pipeline.fit(emptyDataSet)
    }

    assert(caught.getMessage == expectedErrorMessage)
  }

  "A Dependency Parser Approach" should "read CoNLL data from TreeBank format file" in {

    val filesContent = s"Pierre\tNNP\t0${System.lineSeparator()}"+
                        s"is\tVBZ\t1${System.lineSeparator()}"+
                        s"old\tJJ\t2${System.lineSeparator()}${System.lineSeparator()}"+
                        s"Vinken\tNNP\t0${System.lineSeparator()}"+
                        s"is\tVBZ\t1${System.lineSeparator()}"+
                        s"chairman\tNN\t2"
    val dependencyParserApproach = new DependencyParserApproach()
    val expectedResult: List[Sentence] = List(
      List(WordData("Pierre", "NNP", 3), WordData("is", "VBZ", 0), WordData("old", "JJ" ,1)),
      List(WordData("Vinken", "NNP", 3), WordData("is", "VBZ", 0), WordData("chairman", "NN", 1))
    )

    val result = dependencyParserApproach.readCONLL(filesContent)

    assert(expectedResult == result)
  }

  "A Dependency Parser Approach" should "read CoNLL data from universal dependency format file" in {

    val trainingFile = "src/test/resources/parser/labeled/train_small.conllu.txt"
    val externalResource = ExternalResource(trainingFile, ReadAs.LINE_BY_LINE, Map.empty[String, String])
    val conllUAsArray = ResourceHelper.parseLines(externalResource)
    val dependencyParserApproach = new DependencyParserApproach()
    val expectedTrainingSentences: List[Sentence] = List(
      List(WordData("It", "PRON", 2), WordData("should", "AUX", 2), WordData("continue","VERB", 7),
           WordData("to","PART", 5), WordData("be","AUX", 5), WordData("defanged","VERB", 2), WordData(".","PUNCT", 2)),
      List(WordData("So", "ADV", 2), WordData("what", "PRON", 2), WordData("happened","VERB", 4),
           WordData("?","PUNCT", 2)),
      List(WordData("Over", "ADV", 1), WordData("300", "NUM", 2), WordData("Iraqis","PROPN", 4),
        WordData("are","AUX", 4), WordData("reported","VERB", 14), WordData("dead","ADJ", 4),
        WordData("and","CCONJ", 7), WordData("500","NUM", 4), WordData("wounded","ADJ", 7),
        WordData("in","ADP", 10), WordData("Fallujah","PROPN", 4), WordData("alone","ADV", 10),WordData(".","PUNCT", 4)),
      List(WordData("That", "PRON", 3), WordData("too","ADV", 3), WordData("was","AUX", 3),
           WordData("stopped","VERB", 5), WordData(".","PUNCT", 3))
    )

    val trainingSentences = dependencyParserApproach.getTrainingSentencesFromConllU(conllUAsArray)

    assert(expectedTrainingSentences == trainingSentences)

  }


}
