package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.embeddings.WordEmbeddings
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema

import scala.collection.{mutable}

/**
  * Created by jose on 24/11/17.
  */
trait Windowing extends Serializable{

  val before : Int
  val after : Int

  lazy val wordVectors: Option[WordEmbeddings] = None

  /* TODO: create a tokenizer class */
  /* these match the behavior we had when tokenizing sentences for word embeddings */
  val punctuation = Seq(".", ":", ";", ",", "?", "!", "+", "-", "_", "(", ")", "{",
    "}", "#", "mg/kg", "ml", "m2", "cm", "/", "\\", "\"", "'", "[", "]", "%", "<", ">", "&", "=")

  val percent_regex = """([0-9]{1,2}\.[0-9]{1,2}%|[0-9]{1,3}%)"""
  val number_regex = """([0-9]{1,6})"""


  /* apply window, pad/truncate sentence according to window */
  def applyWindow(doc: String, s: Int, e: Int): (Array[String], Array[String], Array[String])  = {

    val target = doc.slice(s, e)
    val targetPart = tokenize(target.trim)
    val leftPart = if (s == 0) Array[String]()
    else tokenize(doc.slice(0, s).trim) //TODO add proper tokenizer here

    val rightPart = if (e == doc.length) Array[String]()
    else tokenize(doc.slice(e, doc.length).trim)

    val (start, leftPadding) =
      if(leftPart.size >= before)
        (leftPart.size - before, Array[String]())
      else
        (0, Array.fill(before - leftPart.length)("empty_marker"))

    val (end, rightPadding) =
      if(targetPart.length - 1 + rightPart.length <= after)
        (rightPart.length, Array.fill(after - (targetPart.length - 1 + rightPart.length))("empty_marker"))
      else
        (after - targetPart.length, Array[String]())

    val leftContext = leftPart.slice(start, leftPart.length)
    val rightContext = rightPart.slice(0, end + 1)

    (leftPadding ++ leftContext, targetPart, rightContext ++ rightPadding)
  }

  /* apply window, pad/truncate sentence according to window */
  def applyWindow(doc: String, target: String) : (Array[String], Array[String], Array[String])= {
    val start = doc.indexOf(target)
    val end = start + target.length
    applyWindow(doc, start, end)
  }

  /* same as above, but convert the resulting text in a vector */
  def applyWindowUdf(wvectors: WordEmbeddings, codes: Map[String, Array[Double]]) =
    udf {(doc:String, pos:mutable.WrappedArray[GenericRowWithSchema], start:Int, end:Int, targetTerm:String)  =>

      val (l, t, r) = applyWindow(doc.toLowerCase, targetTerm.toLowerCase)
      val target = Array.fill(5)(0.2f)
      val nonTarget = Array.fill(5)(0.0f)
      val tmp = l.flatMap(w => wvectors.getEmbeddings(w) ++ nonTarget).map(_.toDouble) ++
        t.flatMap(w => wvectors.getEmbeddings(w) ++ target).map(_.toDouble) ++
        r.flatMap(w => wvectors.getEmbeddings(w) ++ nonTarget).map(_.toDouble)

      Vectors.dense(tmp)

    }

  /* same as above, but convert the resulting text in a vector */
  def applyWindowUdf(wvectors: WordEmbeddings) =
    //here s and e are token number for start and end of target when split on " "
    udf {(doc:String, targetTerm:String, s:Int, e:Int)  =>
      val tokens = doc.split(" ").filter(_!="")

      /* now start and end are indexes in the doc string */
      val start = tokens.slice(0, s).map(_.length).sum +
        tokens.slice(0, s).size // account for spaces
      val end = start + tokens.slice(s, e + 1).map(_.length).sum +
        tokens.slice(s, e + 1).size  // account for spaces

      val (l, t, r) = applyWindow(doc.toLowerCase, start, end)

      var target = Array(0.1, -0.1)
      var nonTarget = Array(-0.1, 0.1)

      val vector : Array[Double] = l.flatMap(w => wvectors.getEmbeddings(w).map(_.toDouble)) ++
      t.flatMap(w =>  wvectors.getEmbeddings(w).map(_.toDouble) ++ normalize(target)) ++
      r.flatMap(w =>  wvectors.getEmbeddings(w).map(_.toDouble) ++ normalize(nonTarget))

      //++ computeLeftDistances(l.takeRight(2), wvectors) ++ computeRightDistances(r.take(2), wvectors)
      if(l.isEmpty || t.isEmpty || r.isEmpty)
        println(vector.sum)

      Vectors.dense(vector)
    }

  val dictWords = Seq("suggest", "evidence", "investigate", "likely", "possibly", "unclear")
    //"possible", "deny", "judge", "father", "history", "appear", "no")
  //"non", "imaging", "consistent", "thought", "prevent", "element")
    //"believe", "rule", "discard", "vs", "either", "may")
  //,"associated", "causes", "leading", "before")

  var dictEmbeddings : Seq[Array[Float]] = Seq()

  import math._

  def distance(xs: Array[Float], ys: Array[Float]):Double = {
    sqrt((xs zip ys).map { case (x,y) => pow(y - x, 2) }.sum)
  }

  def l2norm(xs: Array[Double]):Double = {
    sqrt(xs.map{ x => pow(x, 2)}.sum)
  }

  def normalize(vec: Array[Double]) : Array[Double] = {
    val norm = l2norm(vec) + 1.0
    vec.map(element => element / norm)
  }


  /* original */
  def computeDictDistances(word: String, wvectors: WordEmbeddings) :Array[Double] = {
    val embeddings = dictEmbeddings
    val distances = embeddings.map(e => distance(e, wvectors.getEmbeddings(word))).toArray
    val norm = l2norm(distances)
    distances.map(d => d / norm)
  }

  /*
conditional
  val preComplex = Map("associated" -> List("with"),
    "leading" -> List("to"),
    "history" -> List("of"))

  val preSimple = Array("causes", "caused")

  val postComplex = Map("happens" -> List("when"),
                  "with" -> List("allergies", "stress", "emotional", "climbing"),
                  "on" -> List("climbing"),
                  "at" -> List("work", "home", "rest"))
  */



  val preComplex = Map("suspicious" -> List("for"),
  "could" -> List("be"), "suggestive" -> List("of", "that"),
  "imaging" -> List("for"), "question" -> List("of"),
  "rule" -> List("out"), "evaluation" -> List("for"),
  "attributable" -> List("to"), "consistent" -> List("with"),
  "possibility" -> List("to", "of"), "in" -> List("case"), "for" -> List("presumed"), "with" ->List("possible"))

  val preSimple = Array("suggesting", "suggest", "possible", "presumed", "perhaps", "question", "investigate")

  val posSimple = Array("vs", "or", "occurred")

  val posComplex = Map("was" -> List("considered"))

  def computeLeftDistances(context: Array[String], wvectors: WordEmbeddings):
    Array[Double] = {
    //add distances to single word cues
    val single : Array[Double] = preSimple map { cue =>
      if(context.size > 0)
        distance(wvectors.getEmbeddings(cue), wvectors.getEmbeddings(context.takeRight(1).head))
      else
        -0.5 // we don't know
    }

    //add distances to complex word cues
    val complex = for ((firstToken, secondTokens) <- preComplex;
          secondToken <- secondTokens) yield {
      if(context.size > 1)
        distance(wvectors.getEmbeddings(firstToken), wvectors.getEmbeddings(context.takeRight(2).head)) +
          distance(wvectors.getEmbeddings(secondToken), wvectors.getEmbeddings(context.takeRight(1).head))
      else
        -1 // we don't know
    }
    val distances:Array[Double] = Array(single.min, complex.min)
    normalize(distances.toArray)
  }


  def computeRightDistances(context: Array[String], wvectors: WordEmbeddings):
  Array[Double] = {


    //add distances to complex word cues
    val complex = for ((firstToken, secondTokens) <- posComplex;
                       secondToken <- secondTokens) yield {
      if(context.size > 1)
        distance(wvectors.getEmbeddings(firstToken), wvectors.getEmbeddings(context.head)) +
          distance(wvectors.getEmbeddings(secondToken), wvectors.getEmbeddings(context.tail.head))
      else
        -1.0 // we don't know
    }

    //add distances to single word cues
    val single : Array[Double] = preSimple map { cue =>
      if(context.size > 0)
        distance(wvectors.getEmbeddings(cue), wvectors.getEmbeddings(context.head))
      else
        -0.5 // we don't know
    }

    val distances = Array(single.min, complex.min)
    normalize(distances)
  }


  /* Tokenize a sentence taking care of punctuation */
  def tokenize(sent: String) : Array[String] = {
    var tmp = sent

    // replace percentage
    //tmp = tmp.replaceAll(percent_regex, " percentnum ")

    // replace special characters
    punctuation.foreach(c => tmp = tmp.replace(c, " " + c + " "))

    tmp = tmp.replace(",", " ")
    tmp = tmp.replace("&quot;", "\"")
    tmp = tmp.replace("&apos;", " ' ")


    // replace any number
    // tmp.replaceAll(number_regex, " digitnum ").split(" ").filter(_ != "")
    tmp.split(" ").map(_.trim).filter(_ != "")
  }

}
