package com.johnsnowlabs.ml.ai.util.Generation

case class GenerationConfig(
    bosId: Int,
    padId: Int,
    eosId: Int,
    vocabSize: Int,
    beginSuppressTokens: Option[Array[Int]],
    suppressTokenIds: Option[Array[Int]],
    forcedDecoderIds: Option[Array[(Int, Int)]])
