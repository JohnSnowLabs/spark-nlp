package com.johnsnowlabs.nlp.serialization

import com.github.liblevenshtein.proto.LibLevenshteinProtos.DawgNode
import com.github.liblevenshtein.serialization.PlainTextSerializer
import com.github.liblevenshtein.transducer.{Candidate, ITransducer, Transducer}
import com.johnsnowlabs.nlp.HasFeatures
import com.johnsnowlabs.nlp.annotators.spell.context.parser.SpecialClassParser
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.ConfigLoader
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{Encoder, Encoders, SparkSession}

import scala.reflect.ClassTag

abstract class Feature[Serializable1, Serializable2, TComplete: ClassTag](model: HasFeatures, val name: String) extends Serializable {
  model.features.append(this)

  private val config = ConfigLoader.retrieve
  private val spark = ResourceHelper.spark

  val serializationMode: String = config.getString("sparknlp.settings.annotatorSerializationFormat")
  val useBroadcast: Boolean = config.getBoolean("sparknlp.settings.useBroadcastForFeatures")

  final protected var broadcastValue: Option[Broadcast[TComplete]] = None

  final protected var rawValue: Option[TComplete] = None
  final protected var fallbackRawValue: Option[TComplete] = None

  final protected var fallbackLazyValue: Option[() => TComplete] = None

  final def serialize(spark: SparkSession, path: String, field: String, value: TComplete): Unit = {
    serializationMode match {
      case "dataset" => serializeDataset(spark, path, field, value)
      case "object" => serializeObject(spark, path, field, value)
      case _ => throw new IllegalArgumentException("Illegal performance.serialization setting. Can be 'dataset' or 'object'")
    }
  }

  final def serializeInfer(spark: SparkSession, path: String, field: String, value: Any): Unit =
    serialize(spark, path, field, value.asInstanceOf[TComplete])

  final def deserialize(spark: SparkSession, path: String, field: String): Option[_] = {
    if (broadcastValue.isDefined || rawValue.isDefined)
      throw new Exception(s"Trying de deserialize an already set value for ${this.name}. This should not happen.")
    serializationMode match {
      case "dataset" => deserializeDataset(spark, path, field)
      case "object" => deserializeObject(spark, path, field)
      case _ => throw new IllegalArgumentException("Illegal performance.serialization setting. Can be 'dataset' or 'object'")
    }
  }

  protected def serializeDataset(spark: SparkSession, path: String, field: String, value: TComplete): Unit

  protected def deserializeDataset(spark: SparkSession, path: String, field: String): Option[_]

  protected def serializeObject(spark: SparkSession, path: String, field: String, value: TComplete): Unit

  protected def deserializeObject(spark: SparkSession, path: String, field: String): Option[_]

  final protected def getFieldPath(path: String, field: String): Path =
    Path.mergePaths(new Path(path), new Path("/fields/" + field))

  private def callAndSetFallback: Option[TComplete] = {
    fallbackRawValue = fallbackLazyValue.map(_())
    fallbackRawValue
  }

  final def get: Option[TComplete] = {
    broadcastValue.map(_.value).orElse(rawValue)
  }

  final def orDefault: Option[TComplete] = {
    broadcastValue.map(_.value)
      .orElse(rawValue)
      .orElse(fallbackRawValue)
      .orElse(callAndSetFallback)
  }

  final def getOrDefault: TComplete = {
    orDefault
      .getOrElse(throw new Exception(s"feature $name is not set"))
  }

  final def setValue(value: Option[Any]): HasFeatures = {
    if (useBroadcast) {
      if (isSet) broadcastValue.get.destroy()
      broadcastValue = value.map(v => spark.sparkContext.broadcast[TComplete](v.asInstanceOf[TComplete]))
    } else {
      rawValue = value.map(_.asInstanceOf[TComplete])
    }
    model
  }

  def setFallback(v: Option[() => TComplete]): HasFeatures = {
    fallbackLazyValue = v
    model
  }

  final def isSet: Boolean = {
    broadcastValue.isDefined || rawValue.isDefined
  }

}

class StructFeature[TValue: ClassTag](model: HasFeatures, override val name: String)
  extends Feature[TValue, TValue, TValue](model, name) {

  implicit val encoder: Encoder[TValue] = Encoders.kryo[TValue]

  override def serializeObject(spark: SparkSession, path: String, field: String, value: TValue): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(Seq(value)).saveAsObjectFile(dataPath.toString)
  }

  override def deserializeObject(spark: SparkSession, path: String, field: String): Option[TValue] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.sparkContext.objectFile[TValue](dataPath.toString).first)
    } else {
      None
    }
  }

  override def serializeDataset(spark: SparkSession, path: String, field: String, value: TValue): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.createDataset(Seq(value)).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(spark: SparkSession, path: String, field: String): Option[TValue] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.read.parquet(dataPath.toString).as[TValue].first)
    } else {
      None
    }
  }

}

class MapFeature[TKey: ClassTag, TValue: ClassTag](model: HasFeatures, override val name: String)
  extends Feature[TKey, TValue, Map[TKey, TValue]](model, name) {

  implicit val encoder: Encoder[(TKey, TValue)] = Encoders.kryo[(TKey, TValue)]

  override def serializeObject(spark: SparkSession, path: String, field: String, value: Map[TKey, TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(value.toSeq).saveAsObjectFile(dataPath.toString)
  }



  override def deserializeObject(spark: SparkSession, path: String, field: String): Option[Map[TKey, TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.sparkContext.objectFile[(TKey, TValue)](dataPath.toString).collect.toMap)
    } else {
      None
    }
  }

  override def serializeDataset(spark: SparkSession, path: String, field: String, value: Map[TKey, TValue]): Unit = {
    import spark.implicits._
    val dataPath = getFieldPath(path, field)
    value.toSeq.toDS.write.mode("overwrite").parquet(dataPath.toString)
  }



  override def deserializeDataset(spark: SparkSession, path: String, field: String): Option[Map[TKey, TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.read.parquet(dataPath.toString).as[(TKey, TValue)].collect.toMap)
    } else {
      None
    }
  }

}

class ArrayFeature[TValue: ClassTag](model: HasFeatures, override val name: String)
  extends Feature[TValue, TValue, Array[TValue]](model, name) {

  implicit val encoder: Encoder[TValue] = Encoders.kryo[TValue]

  override def serializeObject(spark: SparkSession, path: String, field: String, value: Array[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(value).saveAsObjectFile(dataPath.toString)
  }

  override def deserializeObject(spark: SparkSession, path: String, field: String): Option[Array[TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.sparkContext.objectFile[TValue](dataPath.toString).collect())
    } else {
      None
    }
  }

  override def serializeDataset(spark: SparkSession, path: String, field: String, value: Array[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.createDataset(value).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(spark: SparkSession, path: String, field: String): Option[Array[TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.read.parquet(dataPath.toString).as[TValue].collect)
    } else {
      None
    }
  }

}

class SetFeature[TValue: ClassTag](model: HasFeatures, override val name: String)
  extends Feature[TValue, TValue, Set[TValue]](model, name) {

  implicit val encoder: Encoder[TValue] = Encoders.kryo[TValue]

  override def serializeObject(spark: SparkSession, path: String, field: String, value: Set[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(value.toSeq).saveAsObjectFile(dataPath.toString)
  }

  override def deserializeObject(spark: SparkSession, path: String, field: String): Option[Set[TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.sparkContext.objectFile[TValue](dataPath.toString).collect().toSet)
    } else {
      None
    }
  }

  override def serializeDataset(spark: SparkSession, path: String, field: String, value: Set[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.createDataset(value.toSeq).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(spark: SparkSession, path: String, field: String): Option[Set[TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.read.parquet(dataPath.toString).as[TValue].collect.toSet)
    } else {
      None
    }
  }

}

class TransducerFeature(model: HasFeatures, override val name: String)
  extends Feature[ITransducer[Candidate], ITransducer[Candidate], ITransducer[Candidate]](model, name) {

  override def serializeObject(spark: SparkSession, path: String, field: String, trans: ITransducer[Candidate]): Unit = {
    val serializer = new PlainTextSerializer
    val dataPath = getFieldPath(path, field)
    val bytes = serializer.serialize(trans)
    spark.sparkContext.parallelize(bytes.toSeq, 1).saveAsObjectFile(dataPath.toString)

  }

  override def deserializeObject(spark: SparkSession, path: String, field: String): Option[ITransducer[Candidate]] = {
    val serializer = new PlainTextSerializer
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      val bytes = spark.sparkContext.objectFile[Byte](dataPath.toString).collect()
      val deserialized = serializer.deserialize(classOf[Transducer[DawgNode, Candidate]], bytes)
      Some(deserialized)
    } else {
      None
    }
  }


  override def serializeDataset(spark: SparkSession, path: String, field: String, trans: ITransducer[Candidate]): Unit = {
    val serializer = new PlainTextSerializer
    import spark.implicits._
    val dataPath = getFieldPath(path, field)
    val bytes = serializer.serialize(trans)
    spark.createDataset(bytes.toSeq).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(spark: SparkSession, path: String, field: String): Option[ITransducer[Candidate]] = {
    val serializer = new PlainTextSerializer
    implicit val encoder: Encoder[Byte] = Encoders.kryo[Byte]
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      val bytes = spark.read.parquet(dataPath.toString).as[Byte].collect
      val deserialized = serializer.deserialize(classOf[Transducer[DawgNode, Candidate]], bytes)
      Some(deserialized)
    } else {
      None
    }
  }
}


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
      elements.foreach{ element =>
          val path = element.getPath()
          if(path.getName.contains("transducer")) {
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
      while(elements.hasNext) {
        val next = elements.next
        val path = next.getPath.toString
        if(path.contains("transducer")) {
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

