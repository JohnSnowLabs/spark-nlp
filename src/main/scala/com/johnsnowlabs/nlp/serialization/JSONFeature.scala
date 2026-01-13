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

import com.johnsnowlabs.nlp.HasFeatures
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.json4s.{DefaultFormats, JValue}

import java.net.URI
import scala.reflect.ClassTag
import scala.util.{Failure, Success, Using}
import org.json4s.native.JsonMethods.parse
import org.json4s.native.Serialization.write

trait JSONFeature {
  this: Feature[_, _, _] =>
  private def fieldPath(path: String, field: String): Path =
    getFieldPath(path, field).suffix("/data.json")

  protected def writeFieldJSON(
      spark: SparkSession,
      path: String,
      field: String,
      jsonSerialized: String): Unit = {
    val (fs, dataPath) = getFieldJSONPath(spark, path, field)
    // ensure parent directories exist
    fs.mkdirs(dataPath.getParent)
    // write JSON to HDFS (overwrite)
    Using(fs.create(dataPath, true)) { out =>
      out.write(jsonSerialized.getBytes("UTF-8"))
      out.flush()
    }
  }

  protected def readFieldJSONString(
      spark: SparkSession,
      path: String,
      field: String): Option[String] = {
    val (fs, dataPath) = getFieldJSONPath(spark, path, field)
    if (fs.exists(dataPath)) {
      // Load the JSON string from the dataPath (read whole file as UTF-8)
      Using(scala.io.Source.fromInputStream(fs.open(dataPath), "UTF-8")) { source =>
        source.mkString
      } match {
        case Success(jsonString) => Some(jsonString)
        case Failure(exception) =>
          logger.error(
            s"Failed to read JSON for field $field at path $dataPath: ${exception.getMessage}")
          None
      }
    } else None
  }

  private def getFieldJSONPath(
      spark: SparkSession,
      path: String,
      field: String): (FileSystem, Path) = {
    val uri = new URI(ResourceHelper.resolvePath(path))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = fieldPath(path, field)
    (fs, dataPath)
  }

  def readFieldJSON(spark: SparkSession, path: String, field: String): Option[JValue] = {
    val (fs, dataPath) = getFieldJSONPath(spark, path, field)
    if (fs.exists(dataPath)) {
      // Load the JSON string from the dataPath (read whole file as UTF-8)
      Using(scala.io.Source.fromInputStream(fs.open(dataPath), "UTF-8")) { source =>
        parse(source.mkString)
      } match {
        case Success(jValue) => Some(jValue)
        case Failure(exception) =>
          logger.error(
            s"Failed to parse JSON for field $field at path $dataPath: ${exception.getMessage}")
          None
      }
    } else None
  }
}

/** Struct Feature for JSON serialization/deserialization.
  *
  * @param model
  *   The parent HasFeatures model
  * @param name
  *   The name of the feature
  * @param jsonSerializer
  *   Function to serialize an instance of TValue to JSON string
  * @param jsonDeserializer
  *   Function to deserialize the value from JSON string and return an instance of TValue
  * @tparam TValue
  *   The type of the value to be serialized/deserialized
  */
class StructJSONFeature[TValue: ClassTag](model: HasFeatures, override val name: String)(
    jsonSerializer: TValue => String,
    jsonDeserializer: String => TValue)
    extends StructFeature[TValue](model, name)
    with JSONFeature {

  override def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      value: TValue): Unit = {
    val jsonSerialized = jsonSerializer(value)
    writeFieldJSON(spark, path, field, jsonSerialized)
  }

  override def deserializeObject(
      spark: SparkSession,
      path: String,
      field: String): Option[TValue] = {
    readFieldJSONString(spark, path, field) match {
      case Some(jsonString) => Some(jsonDeserializer(jsonString))
      case None =>
        super.deserializeObject(spark, path, field)
    }
  }
}

class MapJSONFeature[K: ClassTag, V: ClassTag](model: HasFeatures, override val name: String)(
    jsonSerializer: Map[K, V] => String,
    jsonDeserializer: String => Map[K, V])
    extends MapFeature[K, V](model, name)
    with JSONFeature {

  override def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      value: Map[K, V]): Unit = {
    val jsonSerialized: String = jsonSerializer(value)
    writeFieldJSON(spark, path, field, jsonSerialized)
  }

  override def deserializeObject(
      spark: SparkSession,
      path: String,
      field: String): Option[Map[K, V]] = {
    readFieldJSONString(spark, path, field) match {
      case Some(jsonString) => Some(jsonDeserializer(jsonString))
      case None =>
        super.deserializeObject(spark, path, field)
    }
  }
}
