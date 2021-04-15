package com.johnsnowlabs.nlp.annotators.spell.context.parser

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}
import com.github.liblevenshtein.transducer.factory.TransducerBuilder
import com.github.liblevenshtein.transducer.Algorithm
import com.github.liblevenshtein.serialization.PlainTextSerializer
import com.github.liblevenshtein.transducer.{Candidate, ITransducer}
import com.navigamez.greex.GreexGenerator
import com.johnsnowlabs.nlp.HasFeatures
import com.johnsnowlabs.nlp.serialization.Feature
import com.johnsnowlabs.nlp.annotators.spell.context.WeightedLevenshtein
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.sql.{Encoder, Encoders, SparkSession}

import scala.collection.mutable.Set
import collection.JavaConverters._



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

      spark.sparkContext.parallelize(Seq(specialClass)).
        saveAsObjectFile(s"${dataPath.toString}/${label}")

    }
  }

  override def deserializeObject(spark: SparkSession, path: String, field: String): Option[Seq[SpecialClassParser]] = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    val serializer = new PlainTextSerializer

    if (fs.exists(dataPath)) {
      val elements = fs.listStatus(dataPath)
      var result = Seq[SpecialClassParser]()
      elements.foreach { element =>
        val path = element.getPath()
        val sc = spark.sparkContext.objectFile[SpecialClassParser](path.toString).collect().head
        result = result :+ sc
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


      // the object per se
      spark.createDataset(Seq(specialClass)).
        write.mode("overwrite").
        parquet(s"${dataPath.toString}/${label}")

    }
  }

  override def deserializeDataset(spark: SparkSession, path: String, field: String): Option[Seq[SpecialClassParser]] = {
    import spark.implicits._
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)

    if (fs.exists(dataPath)) {
      val elements = fs.listFiles(dataPath, false)
      var result = Seq[SpecialClassParser]()
      while (elements.hasNext) {
        val next = elements.next
        val path = next.getPath.toString

        // the object
        val sc = spark.read.parquet(path).as[SpecialClassParser].collect.head
        result = result :+ sc
      }
      Some(result)
    } else {
      None
    }
  }
}


trait SpecialClassParser {

  val label:String

  @transient
  var transducer : ITransducer[Candidate] = null
  val maxDist: Int

  def generateTransducer: ITransducer[Candidate]

  def replaceWithLabel(tmp: String): String = {
    if(!transducer.transduce(tmp, 0).iterator.hasNext)
      tmp
    else
      label
  }

  def setTransducer(t: ITransducer[Candidate]) = {
    transducer = t
    this
  }

  def inVocabulary(word:String): Boolean = transducer.transduce(word, 0).iterator.hasNext
}

trait RegexParser extends SpecialClassParser {

  var regex:String

  override def generateTransducer: ITransducer[Candidate] = {

    // first step, enumerate the regular language
    val generator = new GreexGenerator(regex)
    val matches = generator.generateAll.asScala

    // second step, create the transducer
    new TransducerBuilder().
      dictionary(matches.toList.sorted.asJava, true).
      algorithm(Algorithm.STANDARD).
      defaultMaxDistance(maxDist).
      includeDistance(true).
      build[Candidate]
  }

}

trait VocabParser extends SpecialClassParser {

  var vocab: Set[String]

  def generateTransducer: ITransducer[Candidate] = {

    // second step, create the transducer
    new TransducerBuilder().
      dictionary(vocab.toList.sorted.asJava, true).
      algorithm(Algorithm.STANDARD).
      defaultMaxDistance(maxDist).
      includeDistance(true).
      build[Candidate]
  }

  def loadDataset(path:String, col:Option[String] = None) = {
    Set() ++= (scala.io.Source.fromFile(path).getLines)
  }
}

class NumberToken extends RegexParser with SerializableClass {
  /* used during candidate generation(correction) - must be finite */
  override var regex = "([0-9]{1,3}(\\.|,)[0-9]{1,3}|[0-9]{1,2}(\\.[0-9]{1,2})?(%)?|[0-9]{1,4})"
  override val label = "_NUM_"
  override val maxDist: Int = 2

  transducer = generateTransducer

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

  @throws[IOException]
  private def readObject(aInputStream: ObjectInputStream): Unit = {
    transducer = deserializeTransducer(aInputStream)
  }

  @throws[IOException]
  private def writeObject(aOutputStream: ObjectOutputStream): Unit = {
    serializeTransducer(aOutputStream, transducer)
  }
}


class LocationClass() extends VocabParser with SerializableClass {

  override var vocab = Set.empty[String]
  override val label: String = "_LOC_"
  override val maxDist: Int = 3

  def this(path: String) = {
    this()
    vocab = loadDataset(path)
    transducer = generateTransducer
  }

  @throws(classOf[IOException])
  private def readObject(aInputStream: ObjectInputStream): Unit = {
    transducer = deserializeTransducer(aInputStream)
  }

  @throws(classOf[IOException])
  private def writeObject(aOutputStream: ObjectOutputStream): Unit = {
    serializeTransducer(aOutputStream, transducer)
  }
}



class MainVocab() extends VocabParser with SerializableClass {

  override var vocab = Set.empty[String]
  override val label: String = "_MAIN_"
  override val maxDist: Int = 3

  def this(path: String) = {
    this()
    vocab = loadDataset(path)
    transducer = generateTransducer
  }

  @throws(classOf[IOException])
  private def readObject(aInputStream: ObjectInputStream): Unit = {
    transducer = deserializeTransducer(aInputStream)
  }

  @throws(classOf[IOException])
  private def writeObject(aOutputStream: ObjectOutputStream): Unit = {
    serializeTransducer(aOutputStream, transducer)
  }
}


class NamesClass extends VocabParser with SerializableClass {

  override var vocab = Set.empty[String]
  override val label: String = "_NAME_"
  override val maxDist: Int = 3

  def this(path: String) = {
    this()
    vocab = loadDataset(path)
    transducer = generateTransducer
  }

  @throws[IOException]
  private def readObject(aInputStream: ObjectInputStream): Unit = {
    transducer = deserializeTransducer(aInputStream)
  }

  @throws[IOException]
  private def writeObject(aOutputStream: ObjectOutputStream): Unit = {
    serializeTransducer(aOutputStream, transducer)
  }
}



class MedicationClass extends VocabParser with SerializableClass {

  override var vocab = Set.empty[String]
  override val label: String = "_MED_"
  override val maxDist: Int = 3

  def this(path: String) = {
    this()
    vocab = loadDataset(path)
    transducer = generateTransducer
  }

  @throws[IOException]
  private def readObject(aInputStream: ObjectInputStream): Unit = {
    transducer = deserializeTransducer(aInputStream)
  }

  @throws[IOException]
  private def writeObject(aOutputStream: ObjectOutputStream): Unit = {
    serializeTransducer(aOutputStream, transducer)
  }

}

class AgeToken extends RegexParser with SerializableClass {

  override var regex: String = "1?[0-9]{0,2}-(year|month|day)(s)?(-old)?"
  override val label: String = "_AGE_"
  override val maxDist: Int = 2

  transducer = generateTransducer
  @throws[IOException]
  private def readObject(aInputStream: ObjectInputStream): Unit = {
    transducer = deserializeTransducer(aInputStream)
  }

  @throws[IOException]
  private def writeObject(aOutputStream: ObjectOutputStream): Unit = {
    serializeTransducer(aOutputStream, transducer)
  }
}


class UnitToken extends VocabParser with SerializableClass {

  override var vocab: Set[String] = Set("MG=", "MEQ=", "TAB",
    "tablet", "mmHg", "TMIN", "TMAX", "mg/dL", "MMOL/L", "mmol/l", "mEq/L", "mmol/L",
    "mg", "ml", "mL", "mcg", "mcg/", "gram", "unit", "units", "DROP", "intl", "KG", "mcg/inh")
  override val label: String = "_UNIT_"
  override val maxDist: Int = 3

  transducer = generateTransducer

  @throws[IOException]
  private def readObject(aInputStream: ObjectInputStream): Unit = {
    transducer = deserializeTransducer(aInputStream)
  }

  @throws[IOException]
  private def writeObject(aOutputStream: ObjectOutputStream): Unit = {
    serializeTransducer(aOutputStream, transducer)
  }

}

class DateToken extends RegexParser with WeightedLevenshtein with SerializableClass {

  override var regex = "(01|02|03|04|05|06|07|08|09|10|11|12)\\/([0-2][0-9]|30|31)\\/(19|20)[0-9]{2}|[0-9]{2}\\/(19|20)[0-9]{2}|[0-2][0-9]:[0-5][0-9]"
  override val label = "_DATE_"
  override val maxDist: Int = 2

  val dateRegex = "(01|02|03|04|05|06|07|08|09|10|11|12)/[0-3][0-9]/(1|2)[0-9]{3}".r

  transducer = generateTransducer

  def separate(word: String): String = {
    val matcher = dateRegex.pattern.matcher(word)
    if (matcher.matches) {
      word.replace(matcher.group(0), label)
    }
    else
      word
  }

  override def replaceWithLabel(tmp: String): String = separate(tmp)

  @throws[IOException]
  private def readObject(aInputStream: ObjectInputStream): Unit = {
    transducer = deserializeTransducer(aInputStream)
  }

  @throws[IOException]
  private def writeObject(aOutputStream: ObjectOutputStream): Unit = {
    serializeTransducer(aOutputStream, transducer)
  }

}
