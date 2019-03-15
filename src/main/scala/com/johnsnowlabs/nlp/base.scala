package com.johnsnowlabs.nlp

import org.apache.spark.ml.util.DefaultParamsReadable

object base {

  type DocumentAssembler = com.johnsnowlabs.nlp.DocumentAssembler
  object DocumentAssembler extends DefaultParamsReadable[DocumentAssembler]

  type TokenAssembler = com.johnsnowlabs.nlp.TokenAssembler
  object TokenAssembler extends DefaultParamsReadable[TokenAssembler]

  type Doc2Chunk = com.johnsnowlabs.nlp.Doc2Chunk
  object ChunkAssembler extends DefaultParamsReadable[Doc2Chunk]

  type Chunk2Doc = com.johnsnowlabs.nlp.Chunk2Doc
  object Chunk2Doc extends DefaultParamsReadable[Chunk2Doc]

  type Finisher = com.johnsnowlabs.nlp.Finisher
  object Finisher extends DefaultParamsReadable[Finisher]

  type RecursivePipeline = com.johnsnowlabs.nlp.RecursivePipeline

  type LightPipeline = com.johnsnowlabs.nlp.LightPipeline

}
