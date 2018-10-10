package com.johnsnowlabs.nlp.annotators.spell.ocr

import com.github.liblevenshtein.transducer.{Algorithm, Candidate, ITransducer}
import com.github.liblevenshtein.transducer.factory.TransducerBuilder
import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.nlp.annotators.spell.ocr.parser._
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.tensorflow.{Graph, Session}

import scala.collection.mutable

case class OpenClose(open:String, close:String)

class OcrSpellCheckApproach(override val uid: String) extends AnnotatorApproach[OcrSpellCheckModel]{
  override val description: String = "Ocr specific Spell Checking"

  val trainCorpus = new Param[String](this, "trainCorpus", "Path to the training corpus text file.")
  def setTrainCorpus(path: String): this.type = set(trainCorpus, path)

  // TODO make params
  val suffixes = Array(".", ":", "%", ",", ";", "?")
  val prefixes = Array[String]()

  val openClose = Array(OpenClose("(", ")"), OpenClose("[", "]"), OpenClose("(", "),"))

  // Special token classes, TODO: make Params
  val specialClasses = Seq(DateToken, NumberToken)


  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): OcrSpellCheckModel = {

    /* TODO: finish loading the language model */
    val graph = new Graph()
    val config = Array[Byte](56, 1)
    val session = new Session(graph, config)
    val tf = new TensorflowWrapper(session, graph)

    // extract vocabulary
    require(isDefined(trainCorpus), "Train corpus must be set before training")
    val rawText = getOrDefault(trainCorpus)
    val source = scala.io.Source.fromFile(rawText)

    var vocab = mutable.HashMap[String, Double]()
    val firstPass = Seq(SuffixedToken(suffixes ++ openClose.map(_.close)),
      PrefixedToken(openClose.map(_.open)))

    source.getLines.foreach { line =>
      // second pass identify tokens that belong to special classes, and replace with a label
      // TODO removing crazy encodings of space and replacing with standard one
      line.split(" ").flatMap(_.split(" ")).flatMap(_.split(" ")).filter(_!=" ").foreach { token =>
        var tmp = token

        firstPass.foreach{ parser =>
          tmp = parser.separate(tmp)
        }

        specialClasses.foreach { specialClass =>
          tmp = specialClass.replaceWithLabel(tmp)
        }

        tmp.split(" ").foreach {cleanToken =>
           val currCount = vocab.getOrElse(cleanToken, 0.0)
           vocab.update(cleanToken, currCount + 1.0)
        }
      }
    }
    // remove 'rare' tokens, those appearing only one time
    vocab = vocab.filter(_._2 > 1)
    // compute frequencies - logarithmic
    val totalCount = math.log(vocab.values.reduce(_ + _))
    for (key <- vocab.keys){
      vocab.update(key, math.log(vocab(key)) - totalCount)
    }

    // create transducers for special classes
    val specialClassesTransducers = specialClasses.par.map(_.generateTransducer).seq

    // TODO: replace hard coded stuff for development only
    new OcrSpellCheckModel().
      setVocab(vocab).
      setVocabTransducer(createTransducer(vocab.keys.toList)).
      setSpecialClassesTransducers(specialClassesTransducers).
      //setTensorflow(tf).
      //readModel("../auxdata/spell_model", dataset.sparkSession, "").
      setInputCols(getOrDefault(inputCols))
  }

  /* TODO: deprecate (create an expanded list with prefixes and suffixes) */
  private def expList(vocab:List[String]) = {
    val allSuffixes = suffixes ++ openClose.map(_.close)
    val withSuffixes = vocab.toList.flatMap { word =>
      if (!allSuffixes.contains(word))
        Seq(word) ++ suffixes.map(word + _)
      else
        Seq(word)
    }
    val allPrefixes = prefixes ++ openClose.map(_.open)
    val withPrefixes = vocab.toList.flatMap { word =>
      if (!allPrefixes.contains(word))
        allPrefixes.map(_ + word)
      else
        Seq(word)
    }
    withSuffixes ++ withPrefixes
  }


  private def createTransducer(vocab:List[String]) = {
    import scala.collection.JavaConversions._

    // Create Levenshtein Automata
    new TransducerBuilder().
      dictionary(vocab.sorted, true).
      algorithm(Algorithm.STANDARD).
      defaultMaxDistance(2).
      includeDistance(true).
      build[Candidate]
  }

  /* TODO keeping this for reference now, we could relocate it somewhere else */
  private def persistTransducer(transducer:ITransducer[Candidate]) = {
    import com.github.liblevenshtein.serialization.ProtobufSerializer
    import java.nio.file.Files
    import java.nio.file.Paths
    val serializedDictionaryPath = Paths.get("transducer.protobuf.bytes")
    try {
      val stream = Files.newOutputStream(serializedDictionaryPath)
      try {
        val serializer = new ProtobufSerializer
        serializer.serialize(transducer, stream)
      } finally if (stream != null) stream.close()
    }
  }


  def this() = this(Identifiable.randomUID("SPELL"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val requiredAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val annotatorType: AnnotatorType = AnnotatorType.TOKEN

  /* this is a list of functions that return the distance of a string to a particular regex */
  private var tokenClasses = List[(String) => Tuple2[String, Float]]()

}
