package com.johnsnowlabs.nlp.annotators.spell.ocr

import java.io.{BufferedWriter, File, FileWriter}
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

  val trainCorpusPath = new Param[String](this, "trainCorpusPath", "Path to the training corpus text file.")
  def setTrainCorpusPath(path: String): this.type = set(trainCorpusPath, path)

  val vocabPath = new Param[String](this, "vocabPath", "Path to the training corpus text file.")
  def setVocabPath(path: String): this.type = set(vocabPath, path)

  val minCount = new Param[Double](this, "minCount", "Min number of times a token should appear to be included in vocab.")
  def setMinCount(threshold: Double): this.type = set(minCount, threshold)

  setDefault(minCount -> 3.0)

  // TODO make params
  val blackList = Seq("&amp;gt;")
  val suffixes = Array(".", ":", "%", ",", ";", "?", "'")
  val prefixes = Array[String]("'")

  val openClose = Array(OpenClose("(", ")"), OpenClose("[", "]"), OpenClose("\"", "\""))

  private val firstPass = Seq(SuffixedToken(suffixes ++ openClose.map(_.close)),
    PrefixedToken(prefixes ++ openClose.map(_.open)))

  // Special token classes, TODO: make Params
  val specialClasses = Seq(DateToken, NumberToken)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): OcrSpellCheckModel = {

    /* TODO: finish loading the language model */
    val graph = new Graph()
    val config = Array[Byte](56, 1)
    val session = new Session(graph, config)
    val tf = new TensorflowWrapper(session, graph)

    // extract vocabulary
    require(isDefined(trainCorpusPath), "Train corpus must be set before training")
    val rawTextPath = getOrDefault(trainCorpusPath)
    val vPath = getOrDefault(vocabPath)

    val vocab =
      if (new File(vPath).exists())
        loadVocab(vPath).toMap
      else {
        val v = persistVocab(genVocab(rawTextPath), vPath)
        // TODO: analyze v.map(_._1).sorted and see the garbage
        encodeCorpus(rawTextPath, v.map(_._1))
        v.toMap
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

  private def loadVocab(path: String):mutable.HashMap[String, Double] = {
    val vocab = mutable.HashMap[String, Double]()
    scala.io.Source.fromFile(path + ".freq").getLines.foreach { line =>
       val lineFields = line.split("\\|")
      // TODO: this is not working
       vocab += (lineFields(0)-> lineFields(1).toDouble)
    }
    vocab
  }

  def genVocab(rawDataPath: String):List[(String, Double)] = {
    var vocab = mutable.HashMap[String, Double]()

    // TODO: Spark implementation?
    scala.io.Source.fromFile(rawDataPath).getLines.foreach { line =>
      // second pass identify tokens that belong to special classes, and replace with a label
      // TODO removing crazy encodings of space and replacing with standard one
      line.split(" ").flatMap(_.split(" ")).flatMap(_.split(" ")).filter(_!=" ").foreach { token =>
        var tmp = Seq(token)

        firstPass.foreach{ parser =>
          tmp = tmp.flatMap(_.split(" ").map(_.trim)).map(parser.separate).flatMap(_.split(" "))
        }

        specialClasses.foreach { specialClass =>
          tmp = tmp.map(specialClass.replaceWithLabel)
        }

        tmp.foreach {cleanToken =>
          val currCount = vocab.getOrElse(cleanToken, 0.0)
          vocab.update(cleanToken, currCount + 1.0)
        }
      }
    }
    // remove 'rare' tokens, those appearing less than 2 times
    vocab = vocab.filter(_._2 >= getOrDefault(minCount))

    // Blacklists
    // words that appear with first letter capitalized, at the beginning of sentence
    val fwis = vocab.filter(_._1.length > 1).filter(_._1.head.isUpper).
      filter(w => vocab.contains(w._1.head.toLower + w._1.tail)).map(_._1)

    val hyphen = vocab.filter {
      case (word, weight) =>
        val splits = word.split("-")
        splits.length == 2 && vocab.contains(splits(0)) && vocab.contains(splits(1))
    }.map(_._1)

    val slash = vocab.filter {
      case (word, weight) =>
        val splits = word.split("/")
        splits.length == 2 && vocab.contains(splits(0)) && vocab.contains(splits(1))
    }.map(_._1)

    val blacklist = fwis ++ hyphen ++ slash
    blacklist.foreach{vocab.remove}

    // compute frequencies - logarithmic
    val totalCount = math.log(vocab.values.reduce(_ + _))
    for (key <- vocab.keys){
      vocab.update(key, math.log(vocab(key)) - totalCount)
    }

    List(("_PAD_", 0.0), ("_BOS_", 0.0), ("_EOS_", 0.0), ("_UNK_", 0.0)) ++ vocab.toList
  }


  private def persistVocab(v: List[(String, Double)], fileName:String) = {
    // both the vocabulary, and vocabulary + frequencies
    val freqFile = new File(fileName + ".freq")
    val vocabFile = new File(fileName)

    val bwVocabFreq = new BufferedWriter(new FileWriter(freqFile))
    val bwVocab = new BufferedWriter(new FileWriter(vocabFile))

    v.foreach{case (word, freq) =>
      bwVocabFreq.write(s"""$word|$freq""")
      bwVocabFreq.newLine

      bwVocab.write(word)
      bwVocab.newLine
    }
    bwVocab.close
    bwVocabFreq.close
    v
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

  private def encodeCorpus(rawTextPath: String, sorted: List[String]) = {


    val vMap: Map[String, Int] = sorted.zipWithIndex.toMap
    val bw = new BufferedWriter(new FileWriter(new File(rawTextPath + ".ids")))

    scala.io.Source.fromFile(rawTextPath).getLines.foreach { line =>
      // TODO removing crazy encodings of space and replacing with standard one
      val text  = line.split(" ").flatMap(_.split(" ")).flatMap(_.split(" ")).filter(_!=" ").flatMap { token =>
        var tmp = token
        firstPass.foreach{ parser =>
          tmp = parser.separate(tmp)
        }
        // second pass identify tokens that belong to special classes, and replace with a label
        specialClasses.foreach { specialClass =>
          tmp = specialClass.replaceWithLabel(tmp)
        }

        tmp.split(" ").filter(_ != " ").map {cleanToken =>
          s"""${vMap.getOrElse(cleanToken, vMap("_UNK_")).toString}"""
        }
      }.mkString(" ")
      bw.write(s"""${vMap("_BOS_")} $text ${vMap("_EOS_")}\n""")
    }
    bw.close
  }

  def this() = this(Identifiable.randomUID("SPELL"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val requiredAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val annotatorType: AnnotatorType = AnnotatorType.TOKEN

  /* this is a list of functions that return the distance of a string to a particular regex */
  private var tokenClasses = List[(String) => Tuple2[String, Float]]()

}
