package com.johnsnowlabs.ml.ai.model

case class TextEmbeddingResponse(
    `object`: String,
    data: List[EmbeddingData],
    model: String,
    usage: UsageData)

case class EmbeddingData(`object`: String, embedding: List[Float], index: Int)

case class UsageData(prompt_tokens: Int, total_tokens: Int)
