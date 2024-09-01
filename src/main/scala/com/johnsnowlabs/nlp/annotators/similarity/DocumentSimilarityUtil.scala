package com.johnsnowlabs.nlp.annotators.similarity

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.UserDefinedFunction

import scala.util.hashing.MurmurHash3

object DocumentSimilarityUtil {

  import org.apache.spark.sql.functions._

  val mh3Func: String => Int = (s: String) => MurmurHash3.stringHash(s, MurmurHash3.stringSeed)
  val mh3UDF: UserDefinedFunction = udf { mh3Func }

  val averageAggregation: UserDefinedFunction = udf((embeddings: Seq[Seq[Double]]) => {
    val summed = embeddings.transpose.map(_.sum)
    val averaged = summed.map(_ / embeddings.length)
    Vectors.dense(averaged.toArray)
  })

  val firstEmbeddingAggregation: UserDefinedFunction = udf((embeddings: Seq[Seq[Double]]) => {
    Vectors.dense(embeddings.head.toArray)
  })

  val maxAggregation: UserDefinedFunction = udf((embeddings: Seq[Seq[Double]]) => {
    Vectors.dense(embeddings.transpose.map(_.max).toArray)
  })

}
