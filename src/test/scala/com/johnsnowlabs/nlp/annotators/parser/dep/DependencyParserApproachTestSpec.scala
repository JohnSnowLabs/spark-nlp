package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition.{Sentence, WordData}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.SparkSession
import org.scalatest.FlatSpec
import scala.language.existentials


class DependencyParserApproachTestSpec extends FlatSpec{

  private val spark = SparkSession.builder()
    .appName("benchmark")
    .master("local[*]")
    .config("spark.driver.memory", "4G")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  import spark.implicits._

  private val emptyDataSet = spark.createDataset(Seq.empty[String]).toDF("text")

  "A dependency parser that sets TreeBank and CoNLL-U format files " should "raise an error" taggedAs FastTest in {

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

  "A dependency parser that does not set TreeBank or CoNLL-U format files " should "raise an error" taggedAs FastTest in {

    val pipeline = new DependencyParserApproach()
    val expectedErrorMessage = "Either TreeBank or CoNLL-U format file is required."

    val caught = intercept[IllegalArgumentException]{
      pipeline.fit(emptyDataSet)
    }

    assert(caught.getMessage == expectedErrorMessage)
  }

  "A Dependency Parser Approach" should "read CoNLL data from TreeBank format file" taggedAs FastTest in {

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

    val result = dependencyParserApproach.readCONLL(Seq(filesContent.split(System.lineSeparator()).toIterator))

    assert(expectedResult == result)
  }

  "A Dependency Parser Approach" should "read CoNLL data from universal dependency format file" taggedAs FastTest in {

    val trainingFile = "src/test/resources/parser/labeled/train_small.conllu.txt"
    val externalResource = ExternalResource(trainingFile, ReadAs.TEXT, Map.empty[String, String])
    val conllUAsArray = ResourceHelper.parseLines(externalResource)
    val dependencyParserApproach = new DependencyParserApproach()
    val expectedTrainingSentences: List[Sentence] = List(
      List(WordData("It", "PRP", 2), WordData("should", "MD", 2), WordData("continue","VB", 7),
           WordData("to","TO", 5), WordData("be","VB", 5), WordData("defanged","VBN", 2), WordData(".",".", 2)),
      List(WordData("So", "RB", 2), WordData("what", "WP", 2), WordData("happened","VBD", 4),
           WordData("?",".", 2)),
      List(WordData("Over", "RB", 1), WordData("300", "CD", 2), WordData("Iraqis","NNPS", 4),
        WordData("are","VBP", 4), WordData("reported","VBN", 14), WordData("dead","JJ", 4),
        WordData("and","CC", 7), WordData("500","CD", 4), WordData("wounded","JJ", 7),
        WordData("in","IN", 10), WordData("Fallujah","NNP", 4), WordData("alone","RB", 10),WordData(".",".", 4)),
      List(WordData("That", "DT", 3), WordData("too","RB", 3), WordData("was","VBD", 3),
           WordData("stopped","VBN", 5), WordData(".",".", 3))
    )

    val trainingSentences = dependencyParserApproach.getTrainingSentencesFromConllU(conllUAsArray)

    assert(expectedTrainingSentences == trainingSentences)

  }


}
