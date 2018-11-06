package com.johnsnowlabs.nlp.annotators.spell.ocr

import com.github.liblevenshtein.transducer.{Candidate, ITransducer}
import com.johnsnowlabs.ml.tensorflow.{TensorflowSpell, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.assertion.dl.ReadTensorflowModel
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

class OcrSpellCheckModel(override val uid: String) extends AnnotatorModel[OcrSpellCheckModel] with ReadTensorflowModel {

  override val tfFile: String = "good_model"

  private var tensorflow:TensorflowWrapper = null

  private var transducer:ITransducer[Candidate] = null

  private var allTransducers: Seq[ITransducer[Candidate]] = Seq.empty

  private var vocabFreq: Predef.Map[String, Double] = null

  private var vocabIds: Predef.Map[String, Int] = null

  private var idsVocab: Predef.Map[Int, String] = null

  /* TODO get rid of this crap */
  def loadClasses(path:String): Map[Int, (Int, Int)] = {

    scala.io.Source.fromFile(path).getLines.map{line =>
      val chunks = line.split("\\|")
      val key = chunks(0).toInt
      val cid = chunks(1).toInt
      val wcid = chunks(2).toInt
      (key, (cid, wcid))
    }.toMap
  }

  private val classes : Map[Int, (Int, Int)] = loadClasses("classes.psv")

  // the score for the EOS (end of sentence), and BOS (begining of sentence)
  private val eosScore = .01
  private val bosScore = 1.0
  private val gamma = 60.0

  /* limit to the number of candidates we generate for each word */
  private val kBest = 6


  def readModel(path: String, spark: SparkSession, suffix: String): this.type = {
    tensorflow = readTensorflowModel(path, spark, suffix)
    this
  }

  @transient
  private var _model: TensorflowSpell = null

  def model: TensorflowSpell = {
    if (_model == null) {
      require(tensorflow != null, "Tensorflow must be set before usage. Use method setTensorflow() for it.")

      _model = new TensorflowSpell(
        tensorflow,
        Verbose.Silent)
    }
    _model
  }


  def setTensorflow(tf: TensorflowWrapper): OcrSpellCheckModel = {
    tensorflow = tf
    this
  }


  def setVocabTransducer(trans:ITransducer[Candidate]) = {
    transducer = trans
    allTransducers = allTransducers :+ trans
    this
  }

  def setVocabFreq(v: Map[String, Double]) = {
    vocabFreq = v
    this
  }

  def setVocabIds(v: Map[String, Int]) = {
    idsVocab = v.map(_.swap)
    vocabIds = v
    this
  }

  def setSpecialClassesTransducers(transducers: Seq[ITransducer[Candidate]]) = {
    allTransducers = allTransducers ++ transducers
    this
  }

  def decodeViterbi(trellis: Array[Array[(String, Double)]]):(Array[Int], Double) = {

    // delete this
    //val line = Array("_BOS_", "frequently", "black" ,"his", "hemoglobin", "is", "at",  "baseline", ".", "_EOS_").map(vocabIds.get).map(_.get)
    //model.predict(Array(line, line))

    // encode words with ids
    val encTrellis = Array(Array((vocabIds("_BOS_"), bosScore))) ++
          trellis.map(_.map{case (word, weight) =>
            // at this point we keep only those candidates that are in the vocabulary
            (vocabIds.get(word), weight)}.filter(_._1.isDefined).map{case (x,y) => (x.get, y)}) ++
          Array(Array((vocabIds("_EOS_"), eosScore)))

    // init
    var paths = Array(Array(vocabIds("_BOS_")))
    var costs = Array(bosScore) // cost for each of the paths

    for(i <- 1 to encTrellis.length - 1) {

      var newPaths:Array[Array[Int]] = Array()
      var newCosts = Array[Double]()

      val expPaths = encTrellis(i).flatMap{ case (state, _) =>
        paths.map { path =>
          path :+ state
        }
      }

      val cids = expPaths.map(_.map{id => classes.get(id).get._1})
      val cwids = expPaths.map(_.map{id => classes.get(id).get._2})

      val expPathsCosts = model.predict(expPaths, cids, cwids).toArray

      for {((state, wcost), idx) <- encTrellis(i).zipWithIndex} {
        var minCost = Double.MaxValue
        var minPath = Array[Int]()

        val z = (paths, costs).zipped.toList

        for (((path, pathCost), pi) <- z.zipWithIndex) {
          // compute cost to arrive to this 'state' coming from that 'path'
          // val ppl = model.predict(Array(path :+ state))
          val mult = if (i > 1) costs.length else 0
          val ppl = expPathsCosts(idx * mult + pi)

          val cost = pathCost + ppl * wcost
          if (cost < minCost){
            minCost = cost
            minPath = path :+ state
          }
        }
        newPaths = newPaths :+ minPath
        newCosts = newCosts :+ minCost
      }
      paths = newPaths
      costs = newCosts

      paths.zip(costs).foreach{ case (path, cost) =>
          println(path.map(idsVocab.get).map(_.get).toList, cost)

      }
      println("----------")
    }
    // return the path with the lowest cost, and the cost
    paths.zip(costs).minBy(_._2)
  }

  /**
    * takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    import scala.collection.JavaConversions._
    annotations.map{ annotation =>
      val trellis:Array[Array[(String, Double)]] = annotation.result.split(" ").map { token =>
        // ask each token class for candidates, keep the one with lower cost
        val candidates = allTransducers.flatMap(_.transduce(token, 2))
        val min = candidates.map(_.distance).min

        val candW = candidates.map{ c =>
          val weight =  - vocabFreq.getOrElse(c.term, 0.0) / gamma +
                          c.distance.toDouble / token.size
          (c.term, weight)
        }.sortBy(_._2).take(kBest)

        println(s"""$token -> ${candW.toList.take(kBest)}""")
        candW.toArray
      }

      val (decodedPath, cost) = decodeViterbi(trellis)

      annotation.copy(result = decodedPath.map(idsVocab.get).map(_.get).mkString(" "),
        metadata = annotation.metadata + ("score" -> cost.toString))
    }
  }

  def this() = this(Identifiable.randomUID("SPELL"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val requiredAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val annotatorType: AnnotatorType = AnnotatorType.TOKEN

}
