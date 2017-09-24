package com.jsl.nlp.annotators.ner.crf

import java.io._
import java.nio.file.{StandardOpenOption, Paths, Files}

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.rdd.RDD

trait VerboseMode

case class CRFModel(
  head: Array[String],
  dic: Array[(String, Int)],
  alpha: Array[Double]
) {

  case object VerboseLevel1 extends VerboseMode

  case object VerboseLevel2 extends VerboseMode

  protected def formatVersion = "1.0"

  private var verboseMode: Option[VerboseMode] = None

  private var nBest = 0
  private var costFactor = 1.0

  def setNBest(nBest: Int): CRFModel = {
    this.nBest = nBest
    this
  }

  def setVerboseMode(mode: VerboseMode) = {
    this.verboseMode = Some(mode)
    this
  }

  def setcostFact(cf: Double) = {
    this.costFactor = cf
    this
  }

  override def toString: String = {
    val dicString = dic.map { case (k, v) => k + "|-|" + v.toString }
    s"${head.mkString("\t")}|--|${dicString.mkString("\t")}|--|${alpha.map(_.toFloat).mkString("\t")}"
  }

  def toStringHead: String = {
    val dicString: Array[String] = dic.map { case (k, v) => k + "|-|" + v.toString }
    s"${head.mkString("\t")}|--|${dicString.mkString("\t")}"
  }

  def toArrayString: Array[String] = {
    val dicString: Array[String] = dic.map { case (k, v) => k + "|-|" + v.toString }
    val alphaString: Array[String] = alpha.map(_.toString)
    val emptyLine: Array[String] = Array("|--|")
    head ++ emptyLine ++ dicString ++ emptyLine ++ alphaString
  }

  /**
   * Verify CRF model
   *
   * @param tests  Source files to be verified
   * @return Source files with the predictive labels
   */
  def predict(tests: RDD[Sequence]): RDD[Sequence] = {
    val deFeatureIdx = new FeatureIndex()
    deFeatureIdx.readModel(this)
    val bcModel = tests.context.broadcast(this)
    val bcFeature = tests.context.broadcast(deFeatureIdx)
    tests.map(bcModel.value.testCRF(_, bcFeature.value))
  }

  def predict(tests: Array[Sequence]): Seq[Map[String, String]] = {
    val deFeatureIdx = new FeatureIndex()
    deFeatureIdx.readModel(this)
    tests.map(testCRF(_, deFeatureIdx))
    //Todo fix this
    Seq(Map())
  }
  /**
   * Internal method to test the CRF model
   *
   * @param test the sequence to be tested
   * @return the sequence along with predictive labels
   */
  def testCRF(test: Sequence, deFeatureIdx: FeatureIndex): Sequence = {
    val tagger = new Tagger(deFeatureIdx.labels.size, TestMode)
    tagger.setCostFactor(costFactor)
    tagger.setNBest(nBest)
    tagger.read(test, deFeatureIdx)
    deFeatureIdx.buildFeatures(tagger)
    tagger.parse(deFeatureIdx.alpha, verboseMode)
    var Seq: Sequence = null
    if (verboseMode.isDefined) {
      val tokens = new ArrayBuffer[Token]()
      val labels = deFeatureIdx.labels
      val tmp = test.toArray
      for (i <- tmp.indices) {
        val probMat = new ArrayBuffer[(String, Double)]()
        verboseMode match {
          case Some(VerboseLevel1) =>
            probMat.append((labels(tagger.result(i)), tagger.probMatrix(i * labels.length + tagger.result(i))))
          case Some(VerboseLevel2) =>
            for (j <- labels.indices)
              probMat.append((labels(j), tagger.probMatrix(i * labels.length + j)))
          case _ =>
        }
        tokens.append(Token.put(labels(tagger.result(i)), tmp(i).tags).setProb(probMat.toArray))
      }
      Seq = Sequence(tokens.toArray).setSeqProb(tagger.seqProb)
    } else {
      Seq = Sequence(test.toArray.map(x =>
        Token.put(deFeatureIdx.labels(tagger.result(test.toArray.indexOf(x))), x.tags)))
    }
    if (nBest > 0)
      Seq.setCandidates(tagger.topN, tagger.probN, deFeatureIdx.labels)

    Seq
  }
}

object CRFModel {
  def load(source: String): CRFModel = {
    val components = source.split("""\|--\|""")
    require(components.length == 3, "Incompatible formats in Model file")
    val head = components(0).split("\t")
    val dic = components(1).split("\t").map(x => {
      val xx = x.split("""\|-\|""")
      require(xx.length == 2, "Incompatible formats in Model file")
      (xx(0), xx(1).toInt)
    })
    val alpha = components(2).split("\t").map(_.toDouble)
    CRFModel(head, dic, alpha)
  }

  def loadBinaryFile(path: String): CRFModel = {
    val source = scala.io.Source.fromFile(path + "/head").getLines().toArray.head
    val components = source.split("""\|--\|""")
    require(components.length == 2, "Incompatible formats in Model file")
    val head = components(0).split("\t")
    val dic = components(1).split("\t").map(x => {
      val xx = x.split("""\|-\|""")
      require(xx.length == 2, "Incompatible formats in Model file")
      (xx(0), xx(1).toInt)
    })
    val alpha = Array.fill(head(1).toInt)(0.0)
    val infile = new FileInputStream(path + "/alpha")
    val in: DataInputStream = new DataInputStream(infile)
    for (i <- alpha.indices)
      alpha(i) = in.readFloat()
    in.close()
    CRFModel(head, dic, alpha)
  }

  def loadArray(source: Array[String]): CRFModel = {
    val head = new ArrayBuffer[String]()
    val dic = new ArrayBuffer[String]()
    val alpha = new ArrayBuffer[String]()
    var sentinel: Int = 0
    for (line <- source) {
      if (line == "|--|") {
        sentinel += 1
      } else {
        sentinel match {
          case 0 => head.append(line)
          case 1 => dic.append(line)
          case 2 => alpha.append(line)
          case _ => throw new RuntimeException("Incompatible formats in Model")
        }
      }
    }
    CRFModel(head.toArray, dic.toArray.map(x => {
      val xx = x.split("""\|-\|""")
      require(xx.length == 2, "Incompatible formats in Model file")
      (xx(0), xx(1).toInt)
    }), alpha.toArray.map(_.toDouble))
  }

  def save(model: CRFModel): String = {
    model.toString
  }

  def saveBinaryFile(model: CRFModel, path: String) = {
    val head = model.toStringHead
    new java.io.PrintWriter(path + "/head") { write(head); close() }
    val outfile = new FileOutputStream(path + "/alpha")
    val out: DataOutputStream = new DataOutputStream(
      new BufferedOutputStream(
        Files.newOutputStream(
          Paths.get(path + "/alpha"), StandardOpenOption.APPEND
        )
      )
    )
    val alpha = model.alpha.map(_.toFloat)
    for (i <- alpha.indices)
      out.writeFloat(alpha(i))
    out.close()
  }

  def saveArray(model: CRFModel): Array[String] = {
    model.toArrayString
  }
}
