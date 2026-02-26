package com.johnsnowlabs.reader.util

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, explode}

object AssertReaders {

  def assertHierarchy(dataframe: DataFrame, column: String): Unit = {
    val explodedDF = dataframe
      .select(explode(col(column)).as("elem"))
      .select(
        col("elem.elementType").as("elementType"),
        col("elem.content").as("content"),
        col("elem.metadata").as("metadata"))
      .withColumn("element_id", col("metadata")("element_id"))
      .withColumn("parent_id", col("metadata")("parent_id"))
      .cache() // << important to prevent recomputation inconsistencies

    val allElementIds = explodedDF
      .select("element_id")
      .where(col("element_id").isNotNull)
      .distinct()
      .collect()
      .map(_.getString(0))
      .toSet

    val allParentIds = explodedDF
      .select("parent_id")
      .where(col("parent_id").isNotNull)
      .distinct()
      .collect()
      .map(_.getString(0))
      .toSet

    // 1. There should be at least one element with an element_id
    assert(allElementIds.nonEmpty, "No elements have element_id metadata")

    // 2. There should be at least one element with a parent_id
    assert(allParentIds.nonEmpty, "No elements have parent_id metadata")

    // 3. Every parent_id should exist as an element_id
    val missingParents = allParentIds.diff(allElementIds)
    assert(
      missingParents.isEmpty,
      s"Some parent_ids do not correspond to existing element_ids: $missingParents")

    // 4. Each parent should have at least one child
    val parentChildCount = explodedDF
      .filter(col("parent_id").isNotNull)
      .groupBy("parent_id")
      .count()
      .collect()
      .map(r => r.getString(0) -> r.getLong(1))
      .toMap

    assert(
      parentChildCount.nonEmpty && parentChildCount.values.forall(_ >= 1),
      "Each parent_id should have at least one child element")
  }

}
