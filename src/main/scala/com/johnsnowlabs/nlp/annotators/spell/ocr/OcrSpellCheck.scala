package com.johnsnowlabs.nlp.annotators.spell.ocr

import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.nlp.annotators.spell.ocr.parser._
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.tensorflow.{Graph, Session}

import scala.collection.mutable.HashSet

class OcrSpellCheckApproach(override val uid: String) extends AnnotatorApproach[OcrSpellCheckModel]{
  override val description: String = "Ocr specific Spell Checking"

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): OcrSpellCheckModel = {

    // TODO: finish loading the language model
    val graph = new Graph()
    val config = Array[Byte](56, 1)
    val session = new Session(graph, config)
    val tf = new TensorflowWrapper(session, graph)

    // extract vocabulary
    val rawText = "../auxdata/spell_dataset/vocab/spell_corpus.txt" //TODO:should go to param
    val source = scala.io.Source.fromFile(rawText)

    val vocab = HashSet[String]()
    val firstPass = Seq(SuffixedToken, RoundBrackets, DoubleQuotes)

    source.getLines.foreach { line =>

      // first pass separate things
      var tmp = line
      firstPass.foreach{parser =>
        tmp = parser.separate(tmp)
      }

      // second pass identify things, and replace
      tmp.split(" ").filter(_!=" ").foreach { token =>
        vocab += NumberToken.separate(DateToken.separate(token))
      }
    }

    // Create Levenshtein Automata

    // TODO: replace hard coded stuff for development only
    new OcrSpellCheckModel().
      //setTensorflow(tf).
      readModel("../auxdata/spell_model", dataset.sparkSession, "").
      setInputCols(getOrDefault(inputCols))
  }

  def this() = this(Identifiable.randomUID("SPELL"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val requiredAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val annotatorType: AnnotatorType = AnnotatorType.TOKEN

  /* this is a list of functions that return the distance of a string to a particular regex */
  private var tokenClasses = List[(String) => Tuple2[String, Float]]()

}
