package com.johnsnowlabs.nlp.serialization

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.{Encoders, SparkSession}

import scala.reflect.ClassTag


case class SerializationHelper(spark: SparkSession, path: String) {
  import spark.sqlContext.implicits._
  val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

  private def getFieldPath(field: String): Path =
    Path.mergePaths(new Path(path), new Path("/fields/" + field))


  def serializeScalar[TValue: ClassTag](field: String, value: TValue): Unit = {
    implicit val encoder = Encoders.kryo[TValue]

    val dataPath = getFieldPath(field)
    Seq(value).toDS.write.mode("overwrite").parquet(dataPath.toString)
  }

  def deserializeScalar[TValue: ClassTag](field: String): Option[TValue] = {
    implicit val encoder = Encoders.kryo[TValue]

    val dataPath = getFieldPath(field)
    if (fs.exists(dataPath)) {
      val loaded = spark.sqlContext.read.format("parquet").load(dataPath.toString)
      loaded.as[TValue].collect.headOption
    } else {
      None
    }
  }

  def serializeArray[TValue: ClassTag](field: String, value: Array[TValue]): Unit = {
    implicit val encoder = Encoders.kryo[TValue]

    val dataPath = getFieldPath(field)
    value.toSeq.toDS.write.mode("overwrite").parquet(dataPath.toString)
  }

  def deserializeArray[TValue: ClassTag](field: String): Array[TValue] = {
    implicit val encoder = Encoders.kryo[TValue]

    val dataPath = getFieldPath(field)
    if (fs.exists(dataPath)) {
      val loaded = spark.sqlContext.read.format("parquet").load(dataPath.toString)
      loaded.as[TValue].collect
    } else {
      Array.empty
    }
  }

  def serializeMap[TKey: ClassTag, TValue: ClassTag](field: String, value: Map[TKey, TValue]): Unit = {
    implicit val valueEncoder = Encoders.kryo[Map[TKey, TValue]]

    val dataPath = getFieldPath(field)
    Seq(value).toDF().write.mode("overwrite").parquet(dataPath.toString)
  }

  def deserializeMap[TKey: ClassTag, TValue: ClassTag](field: String): Map[TKey, TValue] = {
    implicit val valueEncoder = Encoders.kryo[Map[TKey, TValue]]

    val dataPath = getFieldPath(field)

    if (fs.exists(dataPath)) {
      val loaded = spark.sqlContext.read.format("parquet").load(dataPath.toString)
      loaded.as[Map[TKey, TValue]]
        .collect
        .headOption
        .getOrElse(Map[TKey, TValue]())
    } else {
      Map()
    }
  }
}
