package com.johnsnowlabs.nlp.serialization

import com.johnsnowlabs.nlp.HasFeatures
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.types.{ArrayType, StringType}
import org.apache.spark.sql.{Encoder, Encoders, SparkSession}

import scala.reflect.ClassTag

abstract class Feature[Serializable1, Serializable2, TComplete: ClassTag](model: HasFeatures, val name: String)(implicit val sparkSession: SparkSession = SparkSession.builder().getOrCreate()) extends Serializable {
  model.features.append(this)

  final protected var value: Option[Broadcast[TComplete]] = None

  def serialize(spark: SparkSession, path: String, field: String, value: TComplete): Unit

  final def serializeInfer(spark: SparkSession, path: String, field: String, value: Any): Unit =
    serialize(spark, path, field, value.asInstanceOf[TComplete])

  def deserialize(spark: SparkSession, path: String, field: String): Option[_]

  final protected def getFieldPath(path: String, field: String): Path =
    Path.mergePaths(new Path(path), new Path("/fields/" + field))

  final def get: Option[TComplete] = value.map(_.value)
  final def getValue: TComplete = value.map(_.value).getOrElse(throw new Exception(s"feature $name is not set"))
  final def setValue(v: Option[Any]): HasFeatures = {
    if (isSet) value.get.destroy()
    value = Some(sparkSession.sparkContext.broadcast[TComplete](v.get.asInstanceOf[TComplete]))
    model
  }
  final def isSet: Boolean = value.isDefined

}

class StructFeature[TValue: ClassTag](model: HasFeatures, override val name: String)
  extends Feature[TValue, TValue, TValue](model, name) {

  implicit val encoder: Encoder[TValue] = Encoders.kryo[TValue]

  override def serialize(spark: SparkSession, path: String, field: String, value: TValue): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.createDataset(Seq(value)).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserialize(spark: SparkSession, path: String, field: String): Option[TValue] = {
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

  override def serialize(spark: SparkSession, path: String, field: String, value: Map[TKey, TValue]): Unit = {
    import spark.implicits._
    val dataPath = getFieldPath(path, field)
    value.toSeq.toDS.write.mode("overwrite").parquet(dataPath.toString)
  }



  override def deserialize(spark: SparkSession, path: String, field: String): Option[Map[TKey, TValue]] = {
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

  override def serialize(spark: SparkSession, path: String, field: String, value: Array[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.createDataset(value).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserialize(spark: SparkSession, path: String, field: String): Option[Array[TValue]] = {
    val fs: FileSystem = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.read.parquet(dataPath.toString).as[TValue].collect)
    } else {
      None
    }
  }

}

