package com.johnsnowlabs.nlp.serialization

import com.johnsnowlabs.nlp.HasFeatures
import com.johnsnowlabs.nlp.util.ConfigHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{Encoder, Encoders, SparkSession}

import scala.reflect.ClassTag

abstract class Feature[Serializable1, Serializable2, TComplete: ClassTag](model: HasFeatures, val name: String)(implicit val sparkSession: SparkSession = SparkSession.builder().getOrCreate()) extends Serializable {
  model.features.append(this)

  private val config = ConfigHelper.retrieve

  val serializationMode: String = config.getString("performance.serialization")
  val useBroadcast: Boolean = config.getBoolean("performance.useBroadcast")

  final protected var broadcastValue: Option[Broadcast[TComplete]] = None
  final protected var rawValue: Option[TComplete] = None

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
      throw new Exception(s"Trying de deserialize an already set value for ${this.name}")
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

  final def get: Option[TComplete] = {
    if (useBroadcast)
      broadcastValue.map(_.value)
    else
      rawValue
  }

  final def getValue: TComplete = {
    if (useBroadcast)
      broadcastValue.map(_.value).getOrElse(throw new Exception(s"feature $name is not set"))
    else
      rawValue.getOrElse(throw new Exception(s"feature $name is not set"))
  }

  final def setValue(v: Option[Any]): HasFeatures = {
    if (useBroadcast) {
      if (isSet) broadcastValue.get.destroy()
      broadcastValue = Some(sparkSession.sparkContext.broadcast[TComplete](v.get.asInstanceOf[TComplete]))
    } else {
      rawValue = Some(v.get.asInstanceOf[TComplete])
    }
    model
  }
  final def isSet: Boolean = {
    if (useBroadcast)
      broadcastValue.isDefined
    else
      rawValue.isDefined
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
    val fs: FileSystem = FileSystem.get(spark.sparkContext.hadoopConfiguration)
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
    val fs: FileSystem = FileSystem.get(spark.sparkContext.hadoopConfiguration)
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
    val fs: FileSystem = FileSystem.get(spark.sparkContext.hadoopConfiguration)
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
    val fs: FileSystem = FileSystem.get(spark.sparkContext.hadoopConfiguration)
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
    val fs: FileSystem = FileSystem.get(spark.sparkContext.hadoopConfiguration)
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
    val fs: FileSystem = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.read.parquet(dataPath.toString).as[TValue].collect)
    } else {
      None
    }
  }

}

