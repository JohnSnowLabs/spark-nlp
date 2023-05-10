/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.serialization

import com.github.liblevenshtein.serialization.PlainTextSerializer
import com.johnsnowlabs.nlp.HasFeatures
import com.johnsnowlabs.nlp.annotators.spell.context.parser.VocabParser
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{Encoder, Encoders, SparkSession}

import scala.reflect.ClassTag

abstract class Feature[Serializable1, Serializable2, TComplete: ClassTag](
    model: HasFeatures,
    val name: String)
    extends Serializable {
  model.features.append(this)

  private val spark: SparkSession = ResourceHelper.spark

  val serializationMode: String =
    ConfigLoader.getConfigStringValue(ConfigHelper.serializationMode)
  val useBroadcast: Boolean = ConfigLoader.getConfigBooleanValue(ConfigHelper.useBroadcast)
  final protected var broadcastValue: Option[Broadcast[TComplete]] = None

  final protected var rawValue: Option[TComplete] = None
  final protected var fallbackRawValue: Option[TComplete] = None

  final protected var fallbackLazyValue: Option[() => TComplete] = None
  final protected var isProtected: Boolean = false

  final def serialize(
      spark: SparkSession,
      path: String,
      field: String,
      value: TComplete): Unit = {
    serializationMode match {
      case "dataset" => serializeDataset(spark, path, field, value)
      case "object" => serializeObject(spark, path, field, value)
      case _ =>
        throw new IllegalArgumentException(
          "Illegal performance.serialization setting. Can be 'dataset' or 'object'")
    }
  }

  final def serializeInfer(spark: SparkSession, path: String, field: String, value: Any): Unit =
    serialize(spark, path, field, value.asInstanceOf[TComplete])

  final def deserialize(spark: SparkSession, path: String, field: String): Option[_] = {
    if (broadcastValue.isDefined || rawValue.isDefined)
      throw new Exception(
        s"Trying de deserialize an already set value for ${this.name}. This should not happen.")
    serializationMode match {
      case "dataset" => deserializeDataset(spark, path, field)
      case "object" => deserializeObject(spark, path, field)
      case _ =>
        throw new IllegalArgumentException(
          "Illegal performance.serialization setting. Can be 'dataset' or 'object'")
    }
  }

  protected def serializeDataset(
      spark: SparkSession,
      path: String,
      field: String,
      value: TComplete): Unit

  protected def deserializeDataset(spark: SparkSession, path: String, field: String): Option[_]

  protected def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      value: TComplete): Unit

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
    broadcastValue
      .map(_.value)
      .orElse(rawValue)
      .orElse(fallbackRawValue)
      .orElse(callAndSetFallback)
  }

  final def getOrDefault: TComplete = {
    orDefault
      .getOrElse(throw new Exception(s"feature $name is not set"))
  }

  final def setValue(value: Option[Any]): HasFeatures = {
    //    TODO: make sure we log if there is any protected param is being set
    //     if (isProtected && isSet)
    if (useBroadcast) {
      if (isSet) broadcastValue.get.destroy()
      broadcastValue =
        value.map(v => spark.sparkContext.broadcast[TComplete](v.asInstanceOf[TComplete]))
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

  /** Sets this feature to be protected and only settable once.
    *
    * @return
    *   This Feature
    */
  final def setProtected(): this.type = {
    isProtected = true
    this
  }

}

class StructFeature[TValue: ClassTag](model: HasFeatures, override val name: String)
    extends Feature[TValue, TValue, TValue](model, name) {

  implicit val encoder: Encoder[TValue] = Encoders.kryo[TValue]

  override def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      value: TValue): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(Seq(value)).saveAsObjectFile(dataPath.toString)
  }

  override def deserializeObject(
      spark: SparkSession,
      path: String,
      field: String): Option[TValue] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.sparkContext.objectFile[TValue](dataPath.toString).first)
    } else {
      None
    }
  }

  override def serializeDataset(
      spark: SparkSession,
      path: String,
      field: String,
      value: TValue): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.createDataset(Seq(value)).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(
      spark: SparkSession,
      path: String,
      field: String): Option[TValue] = {
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

  override def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      value: Map[TKey, TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(value.toSeq).saveAsObjectFile(dataPath.toString)
  }

  override def deserializeObject(
      spark: SparkSession,
      path: String,
      field: String): Option[Map[TKey, TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.sparkContext.objectFile[(TKey, TValue)](dataPath.toString).collect.toMap)
    } else {
      None
    }
  }

  override def serializeDataset(
      spark: SparkSession,
      path: String,
      field: String,
      value: Map[TKey, TValue]): Unit = {
    import spark.implicits._
    val dataPath = getFieldPath(path, field)
    value.toSeq.toDS.write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(
      spark: SparkSession,
      path: String,
      field: String): Option[Map[TKey, TValue]] = {
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

  override def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      value: Array[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(value).saveAsObjectFile(dataPath.toString)
  }

  override def deserializeObject(
      spark: SparkSession,
      path: String,
      field: String): Option[Array[TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.sparkContext.objectFile[TValue](dataPath.toString).collect())
    } else {
      None
    }
  }

  override def serializeDataset(
      spark: SparkSession,
      path: String,
      field: String,
      value: Array[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.createDataset(value).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(
      spark: SparkSession,
      path: String,
      field: String): Option[Array[TValue]] = {
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

  override def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      value: Set[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(value.toSeq).saveAsObjectFile(dataPath.toString)
  }

  override def deserializeObject(
      spark: SparkSession,
      path: String,
      field: String): Option[Set[TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.sparkContext.objectFile[TValue](dataPath.toString).collect().toSet)
    } else {
      None
    }
  }

  override def serializeDataset(
      spark: SparkSession,
      path: String,
      field: String,
      value: Set[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.createDataset(value.toSeq).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(
      spark: SparkSession,
      path: String,
      field: String): Option[Set[TValue]] = {
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
    extends Feature[VocabParser, VocabParser, VocabParser](model, name) {

  override def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      trans: VocabParser): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(Seq(trans), 1).saveAsObjectFile(dataPath.toString)

  }

  override def deserializeObject(
      spark: SparkSession,
      path: String,
      field: String): Option[VocabParser] = {
    val serializer = new PlainTextSerializer
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      val sc = spark.sparkContext.objectFile[VocabParser](dataPath.toString).collect().head
      Some(sc)
    } else {
      None
    }
  }

  override def serializeDataset(
      spark: SparkSession,
      path: String,
      field: String,
      trans: VocabParser): Unit = {
    implicit val encoder: Encoder[VocabParser] = Encoders.kryo[VocabParser]
    val serializer = new PlainTextSerializer
    val dataPath = getFieldPath(path, field)
    spark.createDataset(Seq(trans)).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(
      spark: SparkSession,
      path: String,
      field: String): Option[VocabParser] = {
    implicit val encoder: Encoder[VocabParser] = Encoders.kryo[VocabParser]
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      val sc = spark.read.parquet(dataPath.toString).as[VocabParser].collect.head
      Some(sc)
    } else {
      None
    }
  }

}
