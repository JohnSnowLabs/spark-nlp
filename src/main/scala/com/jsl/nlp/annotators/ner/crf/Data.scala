package com.jsl.nlp.annotators.ner.crf

import scala.collection.mutable.ArrayBuffer

/**
 * Class that represents the columns of a token.
 *
 * @param label The last column for this token.
 * @param tags List of tags for this token, expect for the last label.
 */
class Token(
  val label: String,
  val tags: Array[String]
) extends Serializable {
  var prob: Array[(String, Double)] = null

  def setProb(probMat: Array[(String, Double)]): Token = {
    this.prob = probMat
    this
  }

  def probPrinter(): String = {
    val strRes = new StringBuffer()
    strRes.append(tags.mkString("\t"))
    strRes.append("\t" + label + "\t")
    strRes.append(prob.map {
      case (str, p) => str + "/" + p.toString
    }.mkString("\t"))
    strRes.toString
  }

  override def toString: String = {
    s"$label|--|${tags.mkString("|-|")}"
  }

  def compare(other: Token): Int = {
    if (this.label == other.label) 1 else 0
  }
}

object Token {
  /**
   * Parses a string resulted from `LabeledToken#toString` into
   *
   */
  def deSerializer(s: String): Token = {
    val parts = s.split("""\|--\|""")
    val label = parts(0)
    val tags = parts(1).split("""\|-\|""")
    Token.put(label, tags)
  }

  def serializer(token: Token): String = {
    token.toString
  }

  def put(label: String, tags: Array[String]) = {
    new Token(label, tags)
  }

  def put(tags: Array[String]) = {
    new Token(null, tags)
  }
}

/**
 * Class that represents the tokens of a sentence.
 *
 * @param sequence List of tokens
 */
case class Sequence(sequence: Array[Token]) extends Serializable {
  var seqProb = 0.0
  lazy val candidates = ArrayBuffer.empty[Sequence]

  def setSeqProb(seqProb: Double): Sequence = {
    this.seqProb = seqProb
    this
  }

  def setCandidates(
    nBest: ArrayBuffer[Array[Int]],
    probN: ArrayBuffer[Double],
    labels: Array[String]
  ) = {
    for (i <- nBest.indices) {
      val tokens = new ArrayBuffer[Token]()
      for (j <- sequence.indices) {
        tokens += Token.put(labels(nBest(i)(j)), sequence(j).tags)
      }
      candidates += Sequence(tokens.toArray).setSeqProb(probN(i))
    }
    this
  }

  def Print(): String = {
    val strRes = new ArrayBuffer[String]()
    strRes.append("#" + "\t" + seqProb.toString)
    val pairs = this.toArray
    for (i <- pairs.indices) {
      strRes.append(pairs(i).tags.mkString("\t") + "\t" + pairs(i).label)
    }
    strRes.mkString("\n")
  }

  def nthPrint(k: Int): String = {
    val strRes = new ArrayBuffer[String]()
    strRes.append("#" + k + "\t" + candidates(k).seqProb.toString)
    val pairs = this.candidates(k).toArray
    for (i <- pairs.indices) {
      strRes.append(pairs(i).tags.mkString("\t") + "\t" + pairs(i).label)
    }
    strRes.mkString("\n")
  }

  def nBestPrint(): String = {
    val idx = candidates.indices
    idx.map(t => nthPrint(t))
      .mkString("\n")
  }

  override def toString: String = {
    seqProb match {
      case 0.0 => s"${sequence.mkString("\t")}"
      case _ => "#" + seqProb.toString + "\t" + s"${sequence.mkString("\t")}"
    }
  }

  def toArray: Array[Token] = sequence

  def compare(other: Sequence): Int = {
    this.toArray.zip(other.toArray).map { case (one, two) => one.compare(two) }.sum
  }

  def probPrinter(): String = {
    val strRes = new ArrayBuffer[String]()
    strRes.append("|-#-|" + seqProb.toString)
    strRes ++= this.toArray.map(_.probPrinter())
    strRes.mkString("\n")
  }

}

object Sequence {
  def deSerializer(s: String): Sequence = {
    val tokens = s.split("\t")
    tokens.head.substring(0, 5) match {
      case """"\|-#-\|"""" =>
        val seqProb = tokens.head.substring(5).toDouble
        Sequence(tokens.tail.map(Token.deSerializer)).setSeqProb(seqProb)
      case _ => Sequence(tokens.map(Token.deSerializer))
    }
  }
  def serializer(sequence: Sequence): String = {
    sequence.toString
  }
}
