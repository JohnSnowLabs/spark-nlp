package com.johnsnowlabs.nlp.annotators.audio.feature_extractor

object PreprocessorAttributes {
  val Wave2Vec: Seq[String] = Seq(
    "do_normalize",
    "feature_size",
    "padding_side",
    "padding_value",
    "return_attention_mask",
    "sampling_rate")

  val Whisper: Seq[String] = Seq(
    "chunk_length",
    "feature_extractor_type",
    "feature_size",
    "hop_length",
    "n_fft",
    "n_samples",
    "nb_max_frames",
    "padding_side",
    "padding_value",
    "processor_class",
    "return_attention_mask",
    "sampling_rate")
}
