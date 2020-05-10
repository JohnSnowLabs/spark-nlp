package com.johnsnowlabs.nlp.annotators.spell.context.parser

import com.github.liblevenshtein.transducer.factory.TransducerBuilder
import com.github.liblevenshtein.transducer.Algorithm
import com.johnsnowlabs.nlp.serialization.Feature
import com.navigamez.greex.GreexGenerator

import scala.collection.JavaConversions._
import scala.collection.mutable.Set

import com.github.liblevenshtein.proto.LibLevenshteinProtos.DawgNode
import com.github.liblevenshtein.serialization.PlainTextSerializer
import com.github.liblevenshtein.transducer.{Candidate, ITransducer, Transducer}
import com.johnsnowlabs.nlp.HasFeatures
import com.johnsnowlabs.nlp.annotators.spell.context.WeightedLevenshtein
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.sql.{Encoder, Encoders, SparkSession}


class TransducerSeqFeature(model: HasFeatures, override val name: String)
  extends Feature[Seq[SpecialClassParser], Seq[SpecialClassParser], Seq[SpecialClassParser]](model, name) {

  implicit val encoder: Encoder[SpecialClassParser] = Encoders.kryo[SpecialClassParser]

  override def serializeObject(spark: SparkSession, path: String, field: String, specialClasses: Seq[SpecialClassParser]): Unit = {
    import spark.implicits._
    val dataPath = getFieldPath(path, field)
    val serializer = new PlainTextSerializer

    specialClasses.foreach { case specialClass =>

      // hadoop won't see files starting with '_'
      val label = specialClass.label.replaceAll("_", "-")

      val transducer = specialClass.transducer
      specialClass.setTransducer(null)
      // the object per se
      spark.sparkContext.parallelize(Seq(specialClass)).
        saveAsObjectFile(s"${dataPath.toString}/${label}")


      // we handle the transducer separately
      val transBytes = serializer.serialize(transducer)
      spark.sparkContext.parallelize(transBytes.toSeq, 1).
        saveAsObjectFile(s"${dataPath.toString}/${label}transducer")

    }
  }

  override def deserializeObject(spark: SparkSession, path: String, field: String): Option[Seq[SpecialClassParser]] = {
    import scala.collection.JavaConversions._
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    val serializer = new PlainTextSerializer

    if (fs.exists(dataPath)) {
      val elements = fs.listStatus(dataPath)
      var result = Seq[SpecialClassParser]()
      elements.foreach { element =>
        val path = element.getPath()
        if (path.getName.contains("transducer")) {
          // take care of transducer
          val bytes = spark.sparkContext.objectFile[Byte](path.toString).collect()
          val trans = serializer.deserialize(classOf[Transducer[DawgNode, Candidate]], bytes)
          // the object
          val sc = spark.sparkContext.objectFile[SpecialClassParser](path.toString.dropRight(10)).collect().head
          sc.setTransducer(trans)
          result = result :+ sc
        }
      }

      Some(result)
    } else {
      None
    }
  }

  override def serializeDataset(spark: SparkSession, path: String, field: String, specialClasses: Seq[SpecialClassParser]): Unit = {
    implicit val encoder: Encoder[SpecialClassParser] = Encoders.kryo[SpecialClassParser]

    import spark.implicits._
    val dataPath = getFieldPath(path, field)
    specialClasses.foreach { case specialClass =>
      val serializer = new PlainTextSerializer

      // hadoop won't see files starting with '_'
      val label = specialClass.label.replaceAll("_", "-")

      val transducer = specialClass.transducer
      specialClass.setTransducer(null)
      // the object per se
      spark.createDataset(Seq(specialClass)).
        write.mode("overwrite").
        parquet(s"${dataPath.toString}/${label}")

      // we handle the transducer separately
      val transBytes = serializer.serialize(transducer)
      spark.createDataset(transBytes.toSeq).
        write.mode("overwrite").
        parquet(s"${dataPath.toString}/${label}transducer")

    }
  }

  override def deserializeDataset(spark: SparkSession, path: String, field: String): Option[Seq[SpecialClassParser]] = {
    import spark.implicits._
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    val serializer = new PlainTextSerializer

    if (fs.exists(dataPath)) {
      val elements = fs.listFiles(dataPath, false)
      var result = Seq[SpecialClassParser]()
      while (elements.hasNext) {
        val next = elements.next
        val path = next.getPath.toString
        if (path.contains("transducer")) {
          // take care of transducer
          val bytes = spark.read.parquet(path).as[Byte].collect
          val trans = serializer.deserialize(classOf[Transducer[DawgNode, Candidate]], bytes)

          // the object
          val sc = spark.read.parquet(path.dropRight(10)).as[SpecialClassParser].collect.head
          sc.setTransducer(trans)
          result = result :+ sc
        }
      }
      Some(result)
    } else {
      None
    }
  }
}


trait SpecialClassParser {

  val label:String

  var transducer : ITransducer[Candidate]

  val maxDist: Int

  def generateTransducer: ITransducer[Candidate]

  def replaceWithLabel(tmp: String): String = {
    if(transducer.transduce(tmp, 0).toList.isEmpty)
      tmp
    else
      label
  }

  def setTransducer(t: ITransducer[Candidate]) = {
    transducer = t
    this
  }

  def inVocabulary(word:String): Boolean = !transducer.transduce(word, 0).toList.isEmpty
}

trait RegexParser extends SpecialClassParser {

  var regex:String

  override def generateTransducer: ITransducer[Candidate] = {
    import scala.collection.JavaConversions._

    // first step, enumerate the regular language
    val generator = new GreexGenerator(regex)
    val matches = generator.generateAll

    // second step, create the transducer
    new TransducerBuilder().
      dictionary(matches.toList.sorted, true).
      algorithm(Algorithm.STANDARD).
      defaultMaxDistance(maxDist).
      includeDistance(true).
      build[Candidate]
  }

}

trait VocabParser extends SpecialClassParser {

  var vocab: Set[String]

  def generateTransducer: ITransducer[Candidate] = {
    import scala.collection.JavaConversions._

    // second step, create the transducer
    new TransducerBuilder().
      dictionary(vocab.toList.sorted, true).
      algorithm(Algorithm.STANDARD).
      defaultMaxDistance(maxDist).
      includeDistance(true).
      build[Candidate]
  }


  def loadDataset(path:String, col:Option[String] = None) = {
    Set() ++= (scala.io.Source.fromFile(path).getLines)
  }
}

object NumberToken extends RegexParser with Serializable {

  /* used during candidate generation(correction) - must be finite */
  override var regex = "([0-9]{1,3}(\\.|,)[0-9]{1,3}|[0-9]{1,2}(\\.[0-9]{1,2})?(%)?|[0-9]{1,4})"

  override var transducer: ITransducer[Candidate] = generateTransducer

  override val label = "_NUM_"

  override val maxDist: Int = 2

  /* used to parse corpus - potentially infinite */
  private val numRegex =
    """(\-|#|\$)?([0-9]+\.[0-9]+\-[0-9]+\.[0-9]+|[0-9]+/[0-9]+|[0-9]+\-[0-9]+|[0-9]+\.[0-9]+|[0-9]+,[0-9]+|[0-9]+\-[0-9]+\-[0-9]+|[0-9]+)""".r

  def separate(word: String): String = {
    val matcher = numRegex.pattern.matcher(word)
    if(matcher.matches) {
      val result = word.replace(matcher.group(0), label)
      result
    }
    else
      word
  }

  override def replaceWithLabel(tmp: String): String = separate(tmp)

}

// TODO too much repeated code here
class LocationClass extends VocabParser with Serializable {

  override var vocab = Set.empty[String]
  var transducer: ITransducer[Candidate] = null
  override val label: String = "_LOC_"
  override val maxDist: Int = 3

  def this(path: String) = {
    this()
    vocab = loadDataset(path)
    transducer = generateTransducer
  }

}


class NamesClass extends VocabParser with Serializable {

  override var vocab = Set.empty[String]
  var transducer: ITransducer[Candidate] = null
  override val label: String = "_NAME_"
  override val maxDist: Int = 3

  def this(path: String) = {
    this()
    vocab = loadDataset(path)
    transducer = generateTransducer
  }
}


class PossessiveClass extends VocabParser with Serializable {

  override var vocab = Set.empty[String]
  var transducer: ITransducer[Candidate] = null
  override val label: String = "_POSS_"
  override val maxDist: Int = 3

  def this(path: String) = {
    this()
    vocab = loadDataset(path)
    transducer = generateTransducer
  }

}


class MedicationClass extends VocabParser with Serializable {

  override var vocab = Set.empty[String]
  override var transducer: ITransducer[Candidate] = null
  override val label: String = "_MED_"
  override val maxDist: Int = 3

  def this(path: String) = {
    this()
    vocab = loadDataset(path)
    transducer = generateTransducer
  }

}

object AgeToken extends RegexParser with Serializable {

  override var regex: String = "1?[0-9]{0,2}-(year|month|day)(s)?(-old)?"
  override var transducer: ITransducer[Candidate] = generateTransducer
  override val label: String = "_AGE_"
  override val maxDist: Int = 2

}


object UnitToken extends VocabParser with Serializable {

  override var vocab: Set[String] = Set("MG=", "MEQ=", "TAB",
    "tablet", "mmHg", "TMIN", "TMAX", "mg/dL", "MMOL/L", "mmol/l", "mEq/L", "mmol/L",
    "mg", "ml", "mL", "mcg", "mcg/", "gram", "unit", "units", "DROP", "intl", "KG", "mcg/inh")

  override var transducer: ITransducer[Candidate] = generateTransducer
  override val label: String = "_UNIT_"
  override val maxDist: Int = 3

}

object DateToken extends RegexParser with WeightedLevenshtein with Serializable {

  override var regex = "(01|02|03|04|05|06|07|08|09|10|11|12)\\/([0-2][0-9]|30|31)\\/(19|20)[0-9]{2}|[0-9]{2}\\/(19|20)[0-9]{2}|[0-2][0-9]:[0-5][0-9]"
  override var transducer: ITransducer[Candidate] = generateTransducer
  override val label = "_DATE_"
  override val maxDist: Int = 2

  val dateRegex = "(01|02|03|04|05|06|07|08|09|10|11|12)/[0-3][0-9]/(1|2)[0-9]{3}".r

  def separate(word: String): String = {
    val matcher = dateRegex.pattern.matcher(word)
    if (matcher.matches) {
      word.replace(matcher.group(0), label)
    }
    else
      word
  }

  override def replaceWithLabel(tmp: String): String = separate(tmp)

}