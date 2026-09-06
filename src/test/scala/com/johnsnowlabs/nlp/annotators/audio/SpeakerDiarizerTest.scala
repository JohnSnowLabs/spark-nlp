package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.ml.ai.util.Diarization.{ClusteredTurn, SpeakerClustering, SpeakerTurn}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotationAudio, AnnotatorType, AudioAssembler}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

import java.util.concurrent.{Executors, TimeUnit}
import scala.io.Source

class SpeakerDiarizerTest extends AnyFlatSpec with SpeakerDiarizerTestFixtures {

  "windowDuration" should "still correctly separate speakers with a shorter window (5s vs default 10s)" taggedAs SlowTest in {
    val d = freshDiarizer().setWindowDuration(5.0f).setMinDurationOn(0.3f).setMinDurationOff(0.3f)
    val result = run(d, audio("three_speakers"))
    assert(result.nonEmpty, "expected at least one turn")
    assert(
      speakers(result).distinct.length >= 2,
      s"expected >=2 speakers, got ${speakers(result)}")
  }

  "stepDuration" should "still work with a coarser hop (2.0s vs default 1.0s)" taggedAs SlowTest in {
    val d = freshDiarizer().setStepDuration(2.0f).setMinDurationOn(0.3f).setMinDurationOff(0.3f)
    val result = run(d, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    assert(speakers(result).distinct.length >= 1)
  }

  "onsetThreshold" should "detect at least as much speech at a low threshold as at a high one" taggedAs SlowTest in {
    val low = freshDiarizer().setOnsetThreshold(0.2f).setOffsetThreshold(0.2f)
    val high = freshDiarizer().setOnsetThreshold(0.8f).setOffsetThreshold(0.8f)
    val lowResult = run(low, audio("synthetic_2speaker"))
    val highResult = run(high, audio("synthetic_2speaker"))
    val lowTotalMs = lowResult.map(a => a.end - a.begin).sum
    val highTotalMs = highResult.map(a => a.end - a.begin).sum
    assert(
      lowTotalMs >= highTotalMs,
      s"low threshold detected less speech ($lowTotalMs ms) than high threshold ($highTotalMs ms)")
  }

  "minDurationOn" should "filter out short turns when set high" taggedAs SlowTest in {
    val permissive = freshDiarizer().setMinDurationOn(0.0f)
    val strict = freshDiarizer().setMinDurationOn(3.0f)
    val permissiveResult = run(permissive, audio("synthetic_2speaker"))
    val strictResult = run(strict, audio("synthetic_2speaker"))
    assert(
      strictResult.length <= permissiveResult.length,
      s"strict minDurationOn produced more turns (${strictResult.length}) than permissive (${permissiveResult.length})")
    strictResult.foreach(a =>
      assert((a.end - a.begin) >= 3000 - 50, s"turn shorter than 3s survived: $a"))
  }

  "minDurationOff" should "merge turns separated by a gap shorter than this" taggedAs SlowTest in {
    val noMerge = freshDiarizer().setMinDurationOff(0.0f).setMinDurationOn(0.1f)
    val forceMerge = freshDiarizer().setMinDurationOff(30.0f).setMinDurationOn(0.1f)
    val noMergeResult = run(noMerge, audio("synthetic_2speaker"))
    val forceMergeResult = run(forceMerge, audio("synthetic_2speaker"))
    assert(
      forceMergeResult.length <= noMergeResult.length,
      s"forcing a huge merge gap produced more turns (${forceMergeResult.length}) than no merging (${noMergeResult.length})")
  }

  "numSpeakers" should "force exactly 1 cluster when set to 1 on 3-speaker audio" taggedAs SlowTest in {
    val d = freshDiarizer().setNumSpeakers(1).setMinDurationOn(0.3f).setMinDurationOff(0.3f)
    val result = run(d, audio("three_speakers"))
    assert(result.nonEmpty)
    assert(
      speakers(result).distinct.length == 1,
      s"expected 1 speaker, got ${speakers(result).distinct}")
  }

  it should "force exactly 3 clusters when set to 3 on 3-speaker audio" taggedAs SlowTest in {
    val d = freshDiarizer().setNumSpeakers(3).setMinDurationOn(0.3f).setMinDurationOff(0.3f)
    val result = run(d, audio("three_speakers"))
    assert(
      speakers(result).distinct.length == 3,
      s"expected 3 speakers, got ${speakers(result).distinct}")
  }

  "minSpeakers/maxSpeakers" should "cap the number of distinct speakers on 4-speaker audio" taggedAs SlowTest in {
    val d = freshDiarizer()
      .setMinSpeakers(1)
      .setMaxSpeakers(2)
      .setClusteringThreshold(0.01f)
      .setMinDurationOn(0.3f)
      .setMinDurationOff(0.3f)
    val result = run(d, audio("four_speakers"))
    assert(
      speakers(result).distinct.length <= 2,
      s"expected <=2 speakers, got ${speakers(result).distinct}")
  }

  "clusteringThreshold" should "produce fewer or equal distinct speakers when looser than when stricter" taggedAs SlowTest in {
    val loose =
      freshDiarizer().setClusteringThreshold(0.95f).setMinDurationOn(0.3f).setMinDurationOff(0.3f)
    val strict =
      freshDiarizer().setClusteringThreshold(0.05f).setMinDurationOn(0.3f).setMinDurationOff(0.3f)
    val looseResult = run(loose, audio("four_speakers"))
    val strictResult = run(strict, audio("four_speakers"))
    assert(
      speakers(looseResult).distinct.length <= speakers(strictResult).distinct.length,
      s"loose=${speakers(looseResult).distinct.length} strict=${speakers(strictResult).distinct.length}")
  }

  "minSegmentDuration" should "discount confidence for a turn shorter than the threshold" taggedAs SlowTest in {
    val penalized = freshDiarizer().setMinSegmentDuration(0.5f).setMinDurationOn(0.0f)
    val notPenalized = freshDiarizer().setMinSegmentDuration(0.1f).setMinDurationOn(0.0f)
    val shortClip = audio("short_segment")
    val penalizedResult = run(penalized, shortClip)
    val notPenalizedResult = run(notPenalized, shortClip)
    assert(penalizedResult.nonEmpty, "expected the 0.4s clip to still produce a turn")
    assert(notPenalizedResult.nonEmpty)
    val penalizedConf = confidences(penalizedResult).head
    val notPenalizedConf = confidences(notPenalizedResult).head
    assert(
      penalizedConf < notPenalizedConf,
      s"expected penalized confidence ($penalizedConf) < unpenalized ($notPenalizedConf)")
    assert(
      math.abs(penalizedConf - notPenalizedConf * 0.5) < 0.01,
      s"expected exactly a 0.5x penalty: penalized=$penalizedConf unpenalized*0.5=${notPenalizedConf * 0.5}")
  }

  "channelMode=stereo" should "assign one speaker per channel with no ML inference" taggedAs SlowTest in {
    val d = freshDiarizer().setChannelMode("stereo")
    val left = audio("stereo_left")
    val right = audio("stereo_right")
    val row = Array(
      AnnotationAudio(AnnotatorType.AUDIO, left, Map.empty),
      AnnotationAudio(AnnotatorType.AUDIO, right, Map.empty))
    val result = d.batchAnnotate(Seq(row)).head
    assert(
      result.length == 2,
      s"expected exactly 2 annotations (one per channel), got ${result.length}")
    assert(result.map(_.metadata("speaker")).toSet == Set("SPEAKER_00", "SPEAKER_01"))
    assert(result.forall(_.metadata("confidence") == "1.0000"))
  }

  it should "reject an invalid channelMode value" taggedAs SlowTest in {
    assertThrows[IllegalArgumentException] {
      freshDiarizer().setChannelMode("quad")
    }
  }

  "maxChunkDurationSeconds" should "keep the same speaker's label consistent across forced chunk boundaries" taggedAs SlowTest in {
    val d = freshDiarizer()
      .setMaxChunkDurationSeconds(6.0f)
      .setMinDurationOn(0.3f)
      .setMinDurationOff(0.3f)
    val result = run(d, audio("long_multi_chunk"))
    assert(result.nonEmpty)
    val distinctSpeakers = speakers(result).distinct
    println(s"[chunking] turns=${result.length} distinctSpeakers=$distinctSpeakers")
    result.foreach(a =>
      println(s"  [${a.begin / 1000.0}%.2fs-${a.end / 1000.0}%.2fs] ${a.metadata("speaker")}"))
    assert(distinctSpeakers.length >= 3 && distinctSpeakers.length <= 8, s"got $distinctSpeakers")
  }

  "streamingMode" should "keep speaker labels consistent across two separate calls on one machine" taggedAs SlowTest in {
    val sessionId = "test-session-1"
    val d1 = freshDiarizer().setStreamingMode(true).setSessionId(sessionId).setMinDurationOn(0.3f)
    val firstHalf = audio("single_speaker")
    val firstResult = run(d1, firstHalf)
    assert(firstResult.nonEmpty)
    val labelFromFirstCall = firstResult.head.metadata("speaker")

    val d2 = freshDiarizer().setStreamingMode(true).setSessionId(sessionId).setMinDurationOn(0.3f)
    val secondHalf = audio("very_short")
    val secondClip = audio("single_speaker")
    val secondResult = run(d2, secondClip)
    assert(secondResult.nonEmpty)
    assert(
      secondResult.head.metadata("speaker") == labelFromFirstCall,
      s"expected same label across streaming calls: first=$labelFromFirstCall second=${secondResult.head
          .metadata("speaker")}")
  }

  "setStreamingPriorState" should "carry cluster state via an explicit blob rather than the in-memory cache" taggedAs SlowTest in {
    val d1 = freshDiarizer().setMinDurationOn(0.3f)
    val firstResult = run(d1, audio("single_speaker"))
    assert(firstResult.nonEmpty)
    val blob = firstResult.head.metadata("clusterStateSnapshot")
    assert(blob.nonEmpty, "expected a non-empty clusterStateSnapshot on every output annotation")
    val labelFromFirstCall = firstResult.head.metadata("speaker")

    val d2 = freshDiarizer().setMinDurationOn(0.3f).setStreamingPriorState(blob)
    val secondResult = run(d2, audio("single_speaker"))
    assert(secondResult.nonEmpty)
    assert(
      secondResult.head.metadata("speaker") == labelFromFirstCall,
      s"expected same label via explicit state blob: first=$labelFromFirstCall second=${secondResult.head
          .metadata("speaker")}")
  }

  "setSpeakerGallery" should "rename a matching cluster to the enrolled name" taggedAs SlowTest in {
    val probe = freshDiarizer().setMinDurationOn(0.3f).setPersistEmbeddings(true)
    val probeResult = run(probe, audio("synthetic_2speaker"))
    assert(probeResult.nonEmpty)
    val firstTurnEmbedding = probeResult.head.embeddings
    assert(
      firstTurnEmbedding.nonEmpty,
      "expected the output Annotation to carry a real embedding vector")

    val withGallery = freshDiarizer()
      .setMinDurationOn(0.3f)
      .setSpeakerGallery(Map("Enrolled_Speaker" -> firstTurnEmbedding))
    val galleryResult = run(withGallery, audio("synthetic_2speaker"))
    assert(
      galleryResult.head.metadata("speaker") == "Enrolled_Speaker",
      s"expected the matching turn to be renamed, got ${galleryResult.head.metadata("speaker")}")
    assert(
      galleryResult.exists(a => a.metadata("speaker") != "Enrolled_Speaker"),
      "expected the other speaker to remain anonymous, not everyone renamed")
  }

  it should "support remove and clear" taggedAs SlowTest in {
    val d = freshDiarizer()
    d.setSpeakerGallery(Map("A" -> Array(1.0f, 0.0f), "B" -> Array(0.0f, 1.0f)))
    assert(d.getSpeakerGallery.size == 2)
    d.removeSpeakerFromGallery("A")
    assert(d.getSpeakerGallery.size == 1)
    assert(d.getSpeakerGallery.contains("B"))
    d.clearSpeakerGallery()
    assert(d.getSpeakerGallery.isEmpty)
  }

  "persistEmbeddings" should "leave every turn's embeddings empty by default" taggedAs SlowTest in {
    val d = freshDiarizer().setMinDurationOn(0.3f)
    val result = run(d, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    assert(
      result.forall(_.embeddings.isEmpty),
      "a voice embedding is biometric data and should not be emitted unless explicitly opted in")
  }

  it should "populate every turn's embeddings when explicitly opted in" taggedAs SlowTest in {
    val d = freshDiarizer().setMinDurationOn(0.3f).setPersistEmbeddings(true)
    val result = run(d, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    assert(result.forall(_.embeddings.nonEmpty))
  }

  "persistSpeakerGallery" should "NOT persist the gallery by default on save/load" taggedAs SlowTest in {
    val d = freshDiarizer().setSpeakerGallery(Map("Secret" -> Array(1.0f, 2.0f, 3.0f)))
    val path = s"$scratch/models/save_test_no_gallery"
    d.write.overwrite().save(path)
    val reloaded = SpeakerDiarizer.load(path)
    assert(
      reloaded.getSpeakerGallery.isEmpty,
      "gallery should be empty after reload without opting in")
  }

  it should "persist the gallery when explicitly opted in" taggedAs SlowTest in {
    val d = freshDiarizer()
      .setPersistSpeakerGallery(true)
      .setSpeakerGallery(Map("Persisted" -> Array(1.0f, 2.0f, 3.0f)))
    val path = s"$scratch/models/save_test_with_gallery"
    d.write.overwrite().save(path)
    val reloaded = SpeakerDiarizer.load(path)
    assert(
      reloaded.getSpeakerGallery.contains("Persisted"),
      s"gallery after reload: ${reloaded.getSpeakerGallery}")
    assert(reloaded.getSpeakerGallery("Persisted").toSeq == Seq(1.0f, 2.0f, 3.0f))
  }

  "useProfile" should "set the documented call_center defaults" taggedAs SlowTest in {
    val d = freshDiarizer().useProfile("call_center")
    assert(d.getChannelMode == "stereo")
    assert(d.getTranscribe)
    assert(d.getMinSpeakers == 1)
    assert(d.getMaxSpeakers == 2)
  }

  it should "set the documented meeting defaults" taggedAs SlowTest in {
    val d = freshDiarizer().useProfile("meeting")
    assert(d.getChannelMode == "mono")
    assert(d.getTranscribe)
    assert(d.getMinSpeakers == 2)
    assert(d.getMaxSpeakers == 12)
    assert(math.abs(d.getClusteringThreshold - 0.65f) < 1e-6)
  }

  it should "set the documented podcast defaults" taggedAs SlowTest in {
    val d = freshDiarizer().useProfile("podcast")
    assert(d.getChannelMode == "mono")
    assert(d.getTranscribe)
    assert(d.getMinSpeakers == 1)
    assert(d.getMaxSpeakers == 6)
    assert(math.abs(d.getClusteringThreshold - 0.7f) < 1e-6)
  }

  it should "reject an unknown profile name" taggedAs SlowTest in {
    assertThrows[IllegalArgumentException] {
      freshDiarizer().useProfile("bogus_profile")
    }
  }

  it should "still allow individual overrides after applying a preset" taggedAs SlowTest in {
    val d = freshDiarizer().useProfile("meeting").setMaxSpeakers(3)
    assert(d.getMaxSpeakers == 3, "explicit override after useProfile should win")
  }

  "transcribe=true" should "produce real, non-empty transcribed text per turn" taggedAs SlowTest in {
    val d = freshDiarizerWithAsr().setTranscribe(true).setMinDurationOn(0.3f)
    val result = run(d, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    result.foreach { a =>
      println(s"[asr] speaker=${a.metadata("speaker")} text='${a.result}'")
    }
    assert(result.forall(_.result.trim.nonEmpty), "expected every turn to have transcribed text")
    assert(
      result.head.result.trim.split("\\s+").length >= 3,
      s"suspiciously short transcript: '${result.head.result}'")
  }

  it should "produce transcript segments that round-trip through save/load" taggedAs SlowTest in {
    val d = freshDiarizerWithAsr().setTranscribe(true).setMinDurationOn(0.3f)
    val path = s"$scratch/models/save_test_with_asr"
    d.write.overwrite().save(path)
    val reloaded =
      SpeakerDiarizer.load(path).setInputCols("audio_assembler").setOutputCol("speakers")
    assert(reloaded.getTranscribe)
    val result = run(reloaded, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    result.foreach(a =>
      println(s"[asr-reloaded] speaker=${a.metadata("speaker")} text='${a.result}'"))
    assert(
      result.forall(_.result.trim.nonEmpty),
      "expected transcription to still work after save/load")
  }

  it should "throw rather than silently disable transcription when the saved ASR config is corrupted" taggedAs SlowTest in {
    val d = freshDiarizerWithAsr().setTranscribe(true).setMinDurationOn(0.3f)
    val path = s"$scratch/models/save_test_corrupted_asr_config"
    d.write.overwrite().save(path)

    val configFile = new java.io.File(path, "asr_config.json")
    assert(
      configFile.exists(),
      s"expected asr_config.json to exist at $path before corrupting it")
    val writer = new java.io.PrintWriter(configFile)
    try {
      writer.write("{ not valid json at all, corrupted on purpose")
    } finally writer.close()

    assertThrows[Exception] {
      SpeakerDiarizer.load(path)
    }
  }

  it should "load silently with no transcription when no ASR bundle was ever saved at all" taggedAs SlowTest in {
    val d = freshDiarizer().setTranscribe(false).setMinDurationOn(0.3f)
    val path = s"$scratch/models/save_test_no_asr_bundle"
    d.write.overwrite().save(path)

    val configFile = new java.io.File(path, "asr_config.json")
    assert(!configFile.exists(), s"expected no asr_config.json at $path when transcribe=false")

    val reloaded =
      SpeakerDiarizer.load(path).setInputCols("audio_assembler").setOutputCol("speakers")
    val result = run(reloaded, audio("single_speaker"))
    assert(result.nonEmpty)
    assert(
      result.forall(_.result == ""),
      "no ASR bundle was saved, so every result must be empty")
  }

  "transcribe=false" should "leave every turn's result empty" taggedAs SlowTest in {
    val d = freshDiarizer().setTranscribe(false).setMinDurationOn(0.3f)
    val result = run(d, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    assert(result.forall(_.result == ""), s"expected empty result, got ${result.map(_.result)}")
  }

  "silence-only audio" should "produce no turns" taggedAs SlowTest in {
    val d = freshDiarizer()
    val result = run(d, audio("silence_only"))
    assert(result.isEmpty, s"expected empty output for silence, got $result")
  }

  "single-speaker audio" should "produce exactly one distinct speaker label" taggedAs SlowTest in {
    val d = freshDiarizer().setMinDurationOn(0.3f).setMinDurationOff(0.3f)
    val result = run(d, audio("single_speaker"))
    assert(result.nonEmpty)
    assert(
      speakers(result).distinct.length == 1,
      s"expected 1 speaker, got ${speakers(result).distinct}")
  }

  "audio shorter than the segmentation window" should "not crash, even if it produces no turns" taggedAs SlowTest in {
    val d = freshDiarizer()
    val result = run(d, audio("very_short"))
    println(s"[very_short] turns=${result.length}")
  }

  "zero-length audio" should "produce empty output without erroring" taggedAs SlowTest in {
    val d = freshDiarizer()
    val result = run(d, Array.emptyFloatArray)
    assert(result.isEmpty)
  }

  "overlapping speech" should "flag at least one turn as overlap=true" taggedAs SlowTest in {
    val d = freshDiarizer().setMinDurationOn(0.3f).setMinDurationOff(0.3f)
    val result = run(d, audio("overlapping_speech"))
    assert(result.nonEmpty)
    result.foreach(a =>
      println(s"[overlap] [${a.begin / 1000.0}s-${a.end / 1000.0}s] speaker=${a.metadata(
          "speaker")} overlap=${a.metadata("overlap")}"))
    assert(
      result.exists(_.metadata("overlap") == "true"),
      "expected at least one turn flagged as overlapping")
  }

  "SpeakerDiarizer" should "not crash on audio containing NaN and Infinity samples" taggedAs SlowTest in {
    val clean = audio("single_speaker")
    val corrupted = clean.zipWithIndex.map {
      case (_, i) if i % 5000 == 0 => Float.NaN
      case (_, i) if i % 7000 == 0 => Float.PositiveInfinity
      case (_, i) if i % 9000 == 0 => Float.NegativeInfinity
      case (v, _) => v
    }
    val d = freshDiarizer().setMinDurationOn(0.3f)
    val result = run(d, corrupted)
    println(
      s"[NaN/Inf fuzz] turns=${result.length} speakers=${speakers(result).distinct} " +
        s"begins=${result.map(_.begin)} ends=${result.map(_.end)}")
    assert(
      result.forall(a => a.begin >= 0 && a.end >= a.begin),
      "even on corrupted input, begin/end must stay non-negative and ordered")
  }

  it should "produce empty output rather than crashing on all-NaN audio" taggedAs SlowTest in {
    val allNaN = Array.fill(48000)(Float.NaN)
    val d = freshDiarizer().setMinDurationOn(0.3f)
    val result = run(d, allNaN)
    println(s"[all-NaN fuzz] turns=${result.length}")
  }

  "the same audio and params" should "produce byte-identical speaker labels across repeated runs" taggedAs SlowTest in {
    val d1 = freshDiarizer().setMinDurationOn(0.3f)
    val d2 = freshDiarizer().setMinDurationOn(0.3f)
    val r1 = run(d1, audio("three_speakers"))
    val r2 = run(d2, audio("three_speakers"))
    assert(speakers(r1) == speakers(r2), s"non-deterministic: $r1 vs $r2")
    assert(r1.map(a => (a.begin, a.end)) == r2.map(a => (a.begin, a.end)))
  }

  "a plain save/load round trip" should "produce identical output to the original model" taggedAs SlowTest in {
    val original = freshDiarizer().setMinDurationOn(0.3f)
    val path = s"$scratch/models/save_test_plain"
    original.write.overwrite().save(path)
    val reloaded =
      SpeakerDiarizer.load(path).setInputCols("audio_assembler").setOutputCol("speakers")

    val originalResult = run(original, audio("synthetic_2speaker"))
    val reloadedResult = run(reloaded, audio("synthetic_2speaker"))
    assert(speakers(originalResult) == speakers(reloadedResult))
    assert(originalResult.map(a => (a.begin, a.end)) == reloadedResult.map(a => (a.begin, a.end)))
  }

  "offsetThreshold" should "change turn count/duration when varied alone, holding onsetThreshold fixed" taggedAs SlowTest in {
    val lowOffset = freshDiarizer().setOnsetThreshold(0.5f).setOffsetThreshold(0.1f)
    val highOffset = freshDiarizer().setOnsetThreshold(0.5f).setOffsetThreshold(0.9f)
    val lowResult = run(lowOffset, audio("synthetic_2speaker"))
    val highResult = run(highOffset, audio("synthetic_2speaker"))
    val lowTotalMs = lowResult.map(a => a.end - a.begin).sum
    val highTotalMs = highResult.map(a => a.end - a.begin).sum
    println(
      s"[offsetThreshold] low(0.1) totalMs=$lowTotalMs turns=${lowResult.length}; " +
        s"high(0.9) totalMs=$highTotalMs turns=${highResult.length}")
    assert(lowTotalMs >= highTotalMs, s"low=$lowTotalMs high=$highTotalMs")
  }

  "setSpeakerGallery" should "throw a clear error for a wrong-dimension embedding rather than silently misbehaving" taggedAs SlowTest in {
    val d = freshDiarizer()
      .setMinDurationOn(0.3f)
      .setSpeakerGallery(Map("BadDim" -> Array(1.0f, 2.0f, 3.0f)))
    val ex = intercept[IllegalArgumentException] {
      run(d, audio("synthetic_2speaker"))
    }
    println(s"[bad gallery dim] threw as expected: ${ex.getMessage}")
  }

  "loadSavedModel" should "throw a clear error when pointed at a folder missing the ONNX files" taggedAs SlowTest in {
    val emptyDir = s"$scratch/models/empty_model_dir"
    new java.io.File(emptyDir).mkdirs()
    assertThrows[Exception] {
      SpeakerDiarizer.loadSavedModel(emptyDir, spark)
    }
  }

  "stereo mode" should "degrade to fewer speakers rather than crash when given only one channel" taggedAs SlowTest in {
    val d = freshDiarizer().setChannelMode("stereo")
    val row = Array(AnnotationAudio(AnnotatorType.AUDIO, audio("stereo_left"), Map.empty))
    val result = d.batchAnnotate(Seq(row)).head
    assert(
      result.length == 1,
      s"expected exactly 1 annotation for 1 channel, got ${result.length}")
    assert(result.head.metadata("speaker") == "SPEAKER_00")
  }

  "RTTMExporter" should "round-trip real SpeakerDiarizer output, not just synthetic annotations" taggedAs SlowTest in {
    val d = freshDiarizer().setMinDurationOn(0.3f).setMinDurationOff(0.3f)
    val result = run(d, audio("three_speakers"))
    assert(result.nonEmpty)

    val rttmText =
      com.johnsnowlabs.ml.ai.util.Diarization.RTTMExporter.toRTTM(result, "real_test_uri")
    println(s"[rttm] \n$rttmText")
    val parsed = com.johnsnowlabs.ml.ai.util.Diarization.RTTMExporter.fromRTTM(rttmText)

    assert(parsed.length == result.length)
    parsed.zip(result).foreach { case (segment, annotation) =>
      assert(segment.uri == "real_test_uri")
      assert(segment.speaker == annotation.metadata("speaker"))
      assert(math.abs(segment.tbegSeconds - annotation.begin / 1000.0) < 1e-3)
      assert(math.abs(segment.tdurSeconds - (annotation.end - annotation.begin) / 1000.0) < 1e-3)
    }
  }

  "maxChunkDurationSeconds" should "hold up across a much longer clip forced into many more chunks" taggedAs SlowTest in {
    val d = freshDiarizer()
      .setMaxChunkDurationSeconds(15.0f)
      .setMinDurationOn(0.3f)
      .setMinDurationOff(0.3f)
    val clip = audio("very_long")
    val start = System.currentTimeMillis()
    val result = run(d, clip)
    val elapsedSec = (System.currentTimeMillis() - start) / 1000.0
    println(
      s"[large-scale chunking] duration=${clip.length / 16000.0}s turns=${result.length} " +
        s"distinctSpeakers=${speakers(result).distinct} elapsed=${elapsedSec}s")
    assert(result.nonEmpty)
    assert(speakers(result).distinct.length <= 10, s"got ${speakers(result).distinct}")
  }

  "batchAnnotate" should "process multiple independent recordings in one call correctly and independently" taggedAs SlowTest in {
    val d = freshDiarizer().setMinDurationOn(0.3f).setMinDurationOff(0.3f)
    val rows = Seq(
      Array(AnnotationAudio(AnnotatorType.AUDIO, audio("single_speaker"), Map.empty)),
      Array(AnnotationAudio(AnnotatorType.AUDIO, audio("three_speakers"), Map.empty)))
    val results = d.batchAnnotate(rows)
    assert(results.length == 2, "expected one output list per input row")
    val row0Speakers = results(0).map(_.metadata("speaker")).distinct
    val row1Speakers = results(1).map(_.metadata("speaker")).distinct
    println(s"[multi-row] row0=${row0Speakers} row1=${row1Speakers}")
    assert(
      row0Speakers.length == 1,
      s"row 0 (single_speaker) should have 1 speaker, got $row0Speakers")
    assert(
      row1Speakers.length == 3,
      s"row 1 (three_speakers) should have 3 speakers, got $row1Speakers")
  }

  it should "thread cluster state across ROWS within one micro-batch, not just across separate calls" taggedAs SlowTest in {
    val sessionId = "misc-gap-coverage-intra-batch-threading"
    val d = freshDiarizer()
      .setMinDurationOn(0.3f)
      .setStreamingMode(true)
      .setSessionId(sessionId)

    val fullClip = audio("synthetic_2speaker")
    val sampleRate = 16000
    val spkBStart = (7.09 * sampleRate).toInt
    val spkBEnd = (16.11 * sampleRate).toInt
    val speakerBClip = fullClip.slice(spkBStart, spkBEnd)

    val row0 = Array(AnnotationAudio(AnnotatorType.AUDIO, fullClip, Map.empty))
    val row1 = Array(AnnotationAudio(AnnotatorType.AUDIO, speakerBClip, Map.empty))
    val results = d.batchAnnotate(Seq(row0, row1))

    val row0DistinctInOrder = results(0).sortBy(_.begin).map(_.metadata("speaker")).distinct
    assert(
      row0DistinctInOrder.length == 2,
      s"expected 2 speakers in row0 (full 2-speaker clip), got $row0DistinctInOrder")
    val spkBLabelInRow0 = row0DistinctInOrder(1)
    val row1Speakers = speakers(results(1)).distinct
    println(s"[multi-row same-session] row0=$row0DistinctInOrder row1=$row1Speakers")

    assert(
      row1Speakers == Seq(spkBLabelInRow0),
      s"row1 (the same voice as row0's second speaker) must be recognized as the SAME speaker " +
        s"($spkBLabelInRow0) via state threaded across rows of the same micro-batch, not " +
        s"independently renamed to a fresh label, got row1=$row1Speakers")
  }

  "8kHz audio with samplingRate correctly set to 8000" should "still run without crashing, though accuracy is not guaranteed" taggedAs SlowTest in {
    val d = SpeakerDiarizer
      .loadSavedModel(pkgPath, spark)
      .setInputCols("audio_assembler")
      .setOutputCol("speakers")
      .setTranscribe(false)
      .setSamplingRate(8000)
      .setMinDurationOn(0.3f)
      .setMinDurationOff(0.3f)
    val clip8k = loadFloats(s"$td/synthetic_2speaker_8k_floats.txt")
    val result = run(d, clip8k)
    println(s"[8kHz] turns=${result.length} speakers=${speakers(result).distinct} " +
      s"(both models were trained at 16kHz - this documents actual behavior, not a correctness claim)")
  }

  "overlapThreshold" should "flag fewer turns as overlap at a very high threshold than a very low one" taggedAs SlowTest in {
    val strict = freshDiarizer().setMinDurationOn(0.3f).setOverlapThreshold(0.99f)
    val lenient = freshDiarizer().setMinDurationOn(0.3f).setOverlapThreshold(0.01f)

    val strictResult = run(strict, audio("overlapping_speech"))
    val lenientResult = run(lenient, audio("overlapping_speech"))

    def overlapCount(anns: Seq[Annotation]): Int = anns.count(_.metadata("overlap") == "true")

    val strictCount = overlapCount(strictResult)
    val lenientCount = overlapCount(lenientResult)
    println(
      s"[overlapThreshold] strict(0.99)=$strictCount/${strictResult.length} flagged, " +
        s"lenient(0.01)=$lenientCount/${lenientResult.length} flagged")
    assert(
      lenientCount > strictCount,
      s"expected the lenient threshold to flag strictly more turns as overlap, got " +
        s"lenient=$lenientCount strict=$strictCount")
    assert(strictCount == 0, "a 0.99 threshold should be nearly impossible to cross")
  }

  "galleryAcceptanceDistance" should "gate whether a genuinely different speaker gets renamed" taggedAs SlowTest in {
    val probe = freshDiarizer().setMinDurationOn(0.3f).setPersistEmbeddings(true)
    val probeResult = run(probe, audio("three_speakers"))
    val distinctSpeakers = speakers(probeResult).distinct
    assert(
      distinctSpeakers.length >= 2,
      s"expected >=2 distinct speakers from three_speakers.wav, got $distinctSpeakers")
    assert(probeResult.forall(_.embeddings.nonEmpty))

    val turns = probeResult.zipWithIndex.map { case (a, i) =>
      SpeakerTurn(s"probe-turn-$i", a.begin, a.end, a.embeddings)
    }
    val gallery = Map("Enrolled" -> turns.head.embedding)

    val (naturalClustered, _) = SpeakerClustering.cluster(turns, threshold = 0.7)
    val naturalLabelById = naturalClustered.map(t => t.turn.id -> t.speakerLabel).toMap
    val naturalClusterCount = naturalLabelById.values.toSet.size
    assert(naturalClusterCount >= 2, s"expected >=2 natural clusters, got $naturalLabelById")

    def naturalClustersRenamedTo(clustered: Seq[ClusteredTurn], name: String): Set[String] =
      clustered.filter(_.speakerLabel == name).map(ct => naturalLabelById(ct.turn.id)).toSet

    val (strictClustered, _) =
      SpeakerClustering.cluster(
        turns,
        threshold = 0.7,
        gallery = gallery,
        galleryAcceptanceDistance = 0.05)
    val (lenientClustered, _) =
      SpeakerClustering.cluster(
        turns,
        threshold = 0.7,
        gallery = gallery,
        galleryAcceptanceDistance = 2.0)

    val strictRenamed = naturalClustersRenamedTo(strictClustered, "Enrolled")
    val lenientRenamed = naturalClustersRenamedTo(lenientClustered, "Enrolled")

    println(s"[galleryAcceptanceDistance] natural clusters=$naturalClusterCount, " +
      s"strict(0.05) renamed=${strictRenamed.size}, lenient(2.0) renamed=${lenientRenamed.size}")
    assert(
      strictRenamed.size == 1,
      s"only the true matching cluster should be renamed under a strict distance, got $strictRenamed")
    assert(
      lenientRenamed.size == naturalClusterCount,
      s"a 2.0 acceptance distance should accept every natural cluster, got $lenientRenamed of " +
        s"$naturalClusterCount")
    assert(
      lenientRenamed.size > strictRenamed.size,
      "the lenient distance should rename strictly more natural clusters than the strict one")
  }

  "maxEmbeddingClipSeconds" should "still cluster correctly when a long turn is heavily cropped before embedding" taggedAs SlowTest in {
    val capped = freshDiarizer().setMinDurationOn(0.3f).setMaxEmbeddingClipSeconds(2.0f)
    val uncapped =
      freshDiarizer().setMinDurationOn(0.3f)

    val cappedResult = run(capped, audio("single_speaker"))
    val uncappedResult = run(uncapped, audio("single_speaker"))

    println(s"[maxEmbeddingClipSeconds] capped(2.0s) turns=${cappedResult.length} " +
      s"speakers=${speakers(cappedResult).distinct}, uncapped(30.0s) turns=${uncappedResult.length} " +
      s"speakers=${speakers(uncappedResult).distinct}")
    assert(
      cappedResult.nonEmpty && uncappedResult.nonEmpty,
      "cropping must not drop the turn entirely")
    assert(
      speakers(cappedResult).distinct == Seq("SPEAKER_00"),
      "single_speaker.wav is one real speaker - cropping the embedding input must not fragment identity")
    assert(speakers(uncappedResult).distinct == Seq("SPEAKER_00"))
  }

  "maxAsrClipSeconds" should "produce a visibly shorter transcript when capped well below the turn's real length" taggedAs SlowTest in {
    val capped = freshDiarizerWithAsr().setMinDurationOn(0.3f).setMaxAsrClipSeconds(3.0f)
    val uncapped =
      freshDiarizerWithAsr().setMinDurationOn(0.3f)

    val cappedResult = run(capped, audio("single_speaker"))
    val uncappedResult = run(uncapped, audio("single_speaker"))
    assert(cappedResult.nonEmpty && uncappedResult.nonEmpty)

    val cappedLen = cappedResult.map(_.result.length).sum
    val uncappedLen = uncappedResult.map(_.result.length).sum
    println(
      s"[maxAsrClipSeconds] capped(3.0s) chars=$cappedLen text=${cappedResult.map(_.result)}, " +
        s"uncapped(30.0s) chars=$uncappedLen text=${uncappedResult.map(_.result)}")
    assert(
      cappedLen < uncappedLen,
      s"expected transcribing only the first 3s to produce visibly less text than the full " +
        s"~14s turn, got capped=$cappedLen uncapped=$uncappedLen")
  }

  "streamingContextSeconds" should "reduce chunk-boundary fragmentation of one continuous speaker turn" taggedAs SlowTest in {
    val withContext = freshDiarizer()
      .setMinDurationOn(0.3f)
      .setMaxChunkDurationSeconds(4.0f)
      .setStreamingContextSeconds(2.0f)
    val withoutContext = freshDiarizer()
      .setMinDurationOn(0.3f)
      .setMaxChunkDurationSeconds(4.0f)
      .setStreamingContextSeconds(0.0f)

    val withContextResult = run(withContext, audio("single_speaker"))
    val withoutContextResult = run(withoutContext, audio("single_speaker"))

    def describe(anns: Seq[Annotation]): String =
      anns.map(a => f"[${a.begin / 1000.0}%.2fs-${a.end / 1000.0}%.2fs]").mkString(" ")
    println(s"[streamingContextSeconds] withContext(2.0s) turns=${withContextResult.length} " +
      s"${describe(withContextResult)}, withoutContext(0.0s) turns=${withoutContextResult.length} " +
      s"${describe(withoutContextResult)}")
    assert(speakers(withContextResult).distinct == Seq("SPEAKER_00"))
    assert(speakers(withoutContextResult).distinct == Seq("SPEAKER_00"))
    assert(
      withContextResult.length < withoutContextResult.length,
      s"expected real context to stitch boundary-split turns back together, got " +
        s"withContext=${withContextResult.length} withoutContext=${withoutContextResult.length}")

    val withContextSpan = withContextResult.map(_.end).max - withContextResult.map(_.begin).min
    val withoutContextSpan =
      withoutContextResult.map(_.end).max - withoutContextResult.map(_.begin).min
    assert(
      math.abs(withContextSpan - withoutContextSpan) < 500,
      s"stitching should not change the overall covered time span by more than a fraction of a " +
        s"second, got withContext=${withContextSpan}ms withoutContext=${withoutContextSpan}ms")
  }

  "numSpeakers" should "still force an exact split on a complete, non-streaming call" taggedAs SlowTest in {
    val d = freshDiarizer().setMinDurationOn(0.3f).setNumSpeakers(2)
    val result = run(d, audio("single_speaker"))
    println(s"[forceExactSpeakerCount=true, one-shot] speakers=${speakers(result).distinct}")
    assert(
      speakers(result).distinct.length == 2,
      s"a complete one-shot call should still force exactly 2 clusters, got ${speakers(result).distinct}")
  }

  it should "NOT force a phantom split on an early call of an ongoing streaming session" taggedAs SlowTest in {
    val d = freshDiarizer()
      .setMinDurationOn(0.3f)
      .setNumSpeakers(2)
      .setStreamingMode(true)
      .setSessionId("redesign-force-exact-count-test")
    val result = run(d, audio("single_speaker"))
    println(s"[forceExactSpeakerCount=false, streaming] speakers=${speakers(result).distinct}")
    assert(
      speakers(result).distinct.length == 1,
      s"an in-progress streaming call must not manufacture a second speaker out of noise before " +
        s"one has actually been heard, got ${speakers(result).distinct}")
  }

  "a streaming session with numSpeakers set" should "correctly grow from 1 to 2 speakers once a genuinely different voice actually appears" taggedAs SlowTest in {
    val sessionId = "misc-gap-coverage-phantom-lifecycle"
    def freshStreamingDiarizer(): SpeakerDiarizer =
      freshDiarizer()
        .setMinDurationOn(0.3f)
        .setNumSpeakers(2)
        .setStreamingMode(true)
        .setSessionId(sessionId)

    val call1 = run(freshStreamingDiarizer(), audio("single_speaker"))
    val call1Speakers = speakers(call1).distinct
    println(s"[phantom lifecycle] call1 (single_speaker) -> $call1Speakers")
    assert(
      call1Speakers.length == 1,
      s"call 1 should show exactly 1 speaker (no early phantom), got $call1Speakers")

    val fullClip = audio("synthetic_2speaker")
    val sampleRate = 16000
    val spkBStart = (7.09 * sampleRate).toInt
    val spkBEnd = (16.11 * sampleRate).toInt
    val speakerBClip = fullClip.slice(spkBStart, spkBEnd)

    val call2 = run(freshStreamingDiarizer(), speakerBClip)
    val call2Speakers = speakers(call2).distinct
    println(s"[phantom lifecycle] call2 (real 2nd speaker) -> $call2Speakers")

    val allSpeakersSoFar = (call1Speakers ++ call2Speakers).distinct
    println(s"[phantom lifecycle] cumulative session speakers -> $allSpeakersSoFar")
    assert(
      allSpeakersSoFar.length == 2,
      s"the session should now show exactly 2 real speakers (the original one plus the new " +
        s"real one) - neither stuck at 1 (failing to recognize the new voice) nor inflated to " +
        s"3+ (a residual phantom), got $allSpeakersSoFar")
    assert(
      call2Speakers.forall(!call1Speakers.contains(_)),
      s"the new speaker's label(s) in call2 should not collide with call1's, got " +
        s"call1=$call1Speakers call2=$call2Speakers")
  }

  it should "also not force a phantom split via the explicit, multi-executor-safe setStreamingPriorState path" taggedAs SlowTest in {
    val call1 = run(freshDiarizer().setMinDurationOn(0.3f), audio("single_speaker"))
    val call1Speakers = speakers(call1).distinct
    assert(call1Speakers.length == 1, s"expected 1 real speaker in call1, got $call1Speakers")
    val blob = call1.head.metadata("clusterStateSnapshot")

    val call2 = run(
      freshDiarizer().setMinDurationOn(0.3f).setNumSpeakers(2).setStreamingPriorState(blob),
      audio("single_speaker"))
    val call2Speakers = speakers(call2).distinct
    println(s"[explicit prior state, no streamingMode] call1=$call1Speakers call2=$call2Speakers")
    assert(
      call2Speakers.length == 1,
      s"the same real speaker's audio, threaded via an explicit prior state with numSpeakers=2 " +
        s"set, must not be split into a phantom second cluster, got $call2Speakers")
    assert(
      call2Speakers == call1Speakers,
      "it must be recognized as the SAME speaker, not renamed")
  }

  it should "not let a single setStreamingPriorState call leak into a THIRD, unrelated call on the same reused instance" taggedAs SlowTest in {
    val d = freshDiarizer().setMinDurationOn(0.3f)
    val call1 = run(d, audio("single_speaker"))
    val blob = call1.head.metadata("clusterStateSnapshot")
    d.setStreamingPriorState(blob)
    run(d, audio("single_speaker"))

    val call3 = run(d.setNumSpeakers(3), audio("three_speakers"))
    val call3Speakers = speakers(call3).distinct
    println(s"[prior-state leak check] call3 (fresh, unrelated, numSpeakers=3) -> $call3Speakers")
    assert(
      call3Speakers.length == 3,
      s"a call after the prior-state blob should have been fully consumed must still honor an " +
        s"explicit numSpeakers as a fresh one-shot call, got $call3Speakers")
  }

  "transcriptionError metadata" should "be absent from every annotation on a normal, successful transcription" taggedAs SlowTest in {
    val d = freshDiarizerWithAsr().setMinDurationOn(0.3f)
    val result = run(d, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    assert(
      result.forall(a => !a.metadata.contains("transcriptionError")),
      "a successful transcription (even an empty one) must not carry transcriptionError")
  }

  it should "be present when ASR generation genuinely fails" taggedAs SlowTest in {
    val d = freshDiarizerWithAsr().setMinDurationOn(0.3f)
    val model = d.getModelIfNotSet
    val whisperModel = model.whisper.getOrElse(fail("expected a bundled ASR model"))
    val wrappers = whisperModel.onnxWrappers.getOrElse(fail("expected ONNX-backed Whisper"))
    val (encoderSession, _) = wrappers.encoder.getSession(Map.empty)
    encoderSession.close()

    val result = run(d, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    println(
      s"[transcriptionError induced] errors=${result.map(_.metadata.get("transcriptionError"))}")
    assert(
      result.forall(a => a.metadata.get("transcriptionError").exists(_.nonEmpty)),
      "every turn should carry a real transcriptionError message once generation genuinely fails")
    assert(
      result.forall(_.result == ""),
      "the transcript must fall back to empty, not partial/garbage output, on a genuine failure")
  }

  "asrLanguage" should "fall back gracefully to auto-detection for a well-formatted but unsupported code" taggedAs SlowTest in {
    val d = freshDiarizerWithAsr()
      .setMinDurationOn(0.3f)
      .setAsrLanguage("<|zz|>")
    val result = run(d, audio("synthetic_2speaker"))
    println(s"[asrLanguage=<|zz|> unsupported] texts=${result.map(_.result)}")
    assert(result.nonEmpty)
    assert(
      result.exists(_.result.trim.nonEmpty),
      "an unsupported-but-well-formatted language code must not silently produce empty transcripts")
    assert(
      result.forall(a => !a.metadata.contains("transcriptionError")),
      "falling back to auto-detect is not a transcription failure")
  }

  it should "produce the expected real transcript for a genuinely supported code" taggedAs SlowTest in {
    val d = freshDiarizerWithAsr().setMinDurationOn(0.3f).setAsrLanguage("<|en|>")
    val result = run(d, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    assert(
      result.exists(_.result.toLowerCase.contains("she was four years older")),
      s"expected the known ground-truth phrase for this fixture, got ${result.map(_.result)}")
  }

  "setAsrTask/setAsrLanguage" should "be settable and reach the ASR call without erroring" taggedAs SlowTest in {
    val d = freshDiarizerWithAsr()
      .setMinDurationOn(0.3f)
      .setAsrTask("<|transcribe|>")
      .setAsrLanguage("<|en|>")
    val result = run(d, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    assert(
      result.forall(_.result.trim.nonEmpty),
      "expected transcripts with an explicit language/task set")
  }

  it should "reject a malformed asrTask value" taggedAs SlowTest in {
    assertThrows[IllegalArgumentException] {
      freshDiarizer().setAsrTask("bogus")
    }
  }

  it should "reject a malformed asrLanguage value" taggedAs SlowTest in {
    assertThrows[IllegalArgumentException] {
      freshDiarizer().setAsrLanguage("english")
    }
  }

  "toMillis" should "throw rather than silently wrap for a sample index beyond the Int millisecond ceiling" taggedAs SlowTest in {
    val d = freshDiarizer()
    run(d, audio("very_short"))
    val model = d.getModelIfNotSet
    val method = model.getClass.getDeclaredMethod("toMillis", classOf[Long])
    method.setAccessible(true)
    val samplingRate = 16000L
    val hugeSampleIndex: Long = (Int.MaxValue.toLong / 1000 + 10) * samplingRate
    val thrown = intercept[java.lang.reflect.InvocationTargetException] {
      method.invoke(model, java.lang.Long.valueOf(hugeSampleIndex))
    }
    assert(thrown.getCause.isInstanceOf[IllegalArgumentException])
    assert(
      thrown.getCause.getMessage.contains("24.8 days"),
      s"expected a clear overflow message, got: ${thrown.getCause.getMessage}")
  }

  "embeddingModelSize" should "reflect the tier of the loaded model instead of throwing" taggedAs SlowTest in {
    val d = freshDiarizer().setMinDurationOn(0.3f)
    assert(d.getEmbeddingModelSize == "accurate")
  }

  it should "survive a save/load round trip" taggedAs SlowTest in {
    val d = freshDiarizer()
    val path = s"$scratch/models/save_test_embedding_model_size"
    d.write.overwrite().save(path)
    val reloaded = SpeakerDiarizer.load(path)
    assert(reloaded.getEmbeddingModelSize == "accurate")
  }

  "the five new redesign params" should "survive a save/load round trip with non-default values" taggedAs SlowTest in {
    val d = freshDiarizer()
      .setOverlapThreshold(0.42f)
      .setGalleryAcceptanceDistance(0.37f)
      .setPersistEmbeddings(true)
      .setMaxEmbeddingClipSeconds(12.5f)
      .setMaxAsrClipSeconds(17.5f)

    val path = s"$scratch/models/save_test_redesign_params"
    d.write.overwrite().save(path)
    val reloaded = SpeakerDiarizer.load(path)

    assert(reloaded.getOverlapThreshold == 0.42f)
    assert(reloaded.getGalleryAcceptanceDistance == 0.37f)
    assert(reloaded.getPersistEmbeddings)
    assert(reloaded.getMaxEmbeddingClipSeconds == 12.5f)
    assert(reloaded.getMaxAsrClipSeconds == 17.5f)
  }

  "SpeakerDiarizer" should "produce correct, independent results when called from multiple threads concurrently on one broadcasted model" taggedAs SlowTest in {
    val shared = freshDiarizer().setMinDurationOn(0.3f)
    run(shared, audio("very_short"))

    val inputs = Seq(
      "single_speaker" -> 1,
      "synthetic_2speaker" -> 2,
      "three_speakers" -> 2,
      "four_speakers" -> 2)
    val pool = Executors.newFixedThreadPool(inputs.length)
    try {
      val futures = inputs.map { case (name, minExpectedSpeakers) =>
        pool.submit(new java.util.concurrent.Callable[(String, Seq[String])] {
          override def call(): (String, Seq[String]) = {
            val result = run(shared, audio(name))
            (name, speakers(result).distinct)
          }
        })
      }
      val results = futures.map(_.get(120, TimeUnit.SECONDS))
      results.foreach { case (name, distinctSpeakers) =>
        println(s"[concurrency] $name -> $distinctSpeakers")
      }
      val expectedMinBy = inputs.toMap
      results.foreach { case (name, distinctSpeakers) =>
        assert(
          distinctSpeakers.length >= expectedMinBy(name),
          s"$name: expected >=${expectedMinBy(name)} distinct speakers under concurrent " +
            s"execution, got $distinctSpeakers - a real cross-thread contamination bug would " +
            s"most plausibly show up as the wrong speaker count here")
      }
      val sequential = inputs.map { case (name, _) =>
        name -> speakers(run(shared, audio(name))).distinct
      }
      assert(
        results.toMap.mapValues(_.length) == sequential.toMap.mapValues(_.length),
        s"concurrent and sequential runs disagreed on speaker counts: " +
          s"concurrent=${results.toMap.mapValues(_.length)} sequential=${sequential.toMap
              .mapValues(_.length)}")
    } finally {
      pool.shutdown()
    }
  }

  "setMaxOutputLength" should "actually truncate transcripts when set very small" taggedAs SlowTest in {
    val short = freshDiarizerWithAsr().setMinDurationOn(0.3f).setMaxOutputLength(8)
    val result = run(short, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    val shortLen = result.map(_.result.length).sum
    println(s"[maxOutputLength=8] totalChars=$shortLen texts=${result.map(_.result)}")
    assert(shortLen < 200, s"expected a heavily truncated transcript, got $shortLen chars total")
  }

  "setDoSample/setTemperature" should "be reachable without erroring (sampled generation)" taggedAs SlowTest in {
    val sampled = freshDiarizerWithAsr()
      .setMinDurationOn(0.3f)
      .setDoSample(true)
      .setTemperature(0.8)
      .setRandomSeed(42L)
    val result = run(sampled, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    result.foreach(a => println(s"[doSample=true] ${a.metadata("speaker")}: ${a.result}"))
  }

  "greedy decoding (doSample=false, the default)" should "be deterministic across repeated ASR runs" taggedAs SlowTest in {
    val d1 = freshDiarizerWithAsr().setMinDurationOn(0.3f)
    val r1 = run(d1, audio("synthetic_2speaker"))
    val d2 = freshDiarizerWithAsr().setMinDurationOn(0.3f)
    val r2 = run(d2, audio("synthetic_2speaker"))
    assert(r1.map(_.result) == r2.map(_.result), s"ASR text should be deterministic: $r1 vs $r2")
  }

  "minOutputLength" should "force a visibly longer transcript than the natural, unconstrained length" taggedAs SlowTest in {
    val default = run(freshDiarizerAsrDefault(), audio("synthetic_2speaker"))
    val forced =
      run(freshDiarizerAsrDefault().setMinOutputLength(200), audio("synthetic_2speaker"))

    val defaultLen = text(default).length
    val forcedLen = text(forced).length
    println(s"[minOutputLength] default(0) chars=$defaultLen, forced(200) chars=$forcedLen")
    assert(
      forcedLen > defaultLen,
      s"forcing minOutputLength=200 should suppress EOS well past the natural stopping point, " +
        s"got default=$defaultLen forced=$forcedLen")
  }

  "topK" should "silently floor to TopKLogitWarper's hardcoded minTokensToKeep=100 for small values" taggedAs SlowTest in {
    val topK1 = run(
      freshDiarizerAsrDefault().setDoSample(true).setRandomSeed(777L).setTopK(1),
      audio("synthetic_2speaker"))
    val topK99 = run(
      freshDiarizerAsrDefault().setDoSample(true).setRandomSeed(777L).setTopK(99),
      audio("synthetic_2speaker"))
    val topK5000 = run(
      freshDiarizerAsrDefault().setDoSample(true).setRandomSeed(777L).setTopK(5000),
      audio("synthetic_2speaker"))

    println(
      s"[topK floor] topK=1: ${text(topK1)} | topK=99: ${text(topK99)} | topK=5000: ${text(topK5000)}")
    assert(
      text(topK1) == text(topK99),
      "topK=1 and topK=99 both floor to the same effective 100-candidate pool and must match " +
        "exactly under the same seed")
  }

  "topP" should "collapse to greedy-equivalent output at a near-zero value" taggedAs SlowTest in {
    val greedy = run(freshDiarizerAsrDefault(), audio("synthetic_2speaker"))
    val tinyTopP = run(
      freshDiarizerAsrDefault().setDoSample(true).setRandomSeed(42L).setTopP(0.0001),
      audio("synthetic_2speaker"))

    println(s"[topP~0] greedy=${text(greedy)} | topP=0.0001=${text(tinyTopP)}")
    assert(
      text(greedy) == text(tinyTopP),
      "a near-zero topP should admit essentially only the single highest-probability token at " +
        "each step, matching greedy decoding")
  }

  private lazy val repeatedSpeech: Array[Float] = {
    val clip = audio("single_speaker")
    clip ++ clip
  }

  "repetitionPenalty" should "change greedy output on speech that verbatim-repeats itself" taggedAs SlowTest in {
    val baseline =
      run(freshDiarizerAsrDefault().setRepetitionPenalty(1.0), repeatedSpeech)
    val penalized =
      run(freshDiarizerAsrDefault().setRepetitionPenalty(1.8), repeatedSpeech)

    println(s"[repetitionPenalty] baseline=${text(baseline)} | penalized=${text(penalized)}")
    assert(text(baseline).nonEmpty && text(penalized).nonEmpty)
    assert(
      text(baseline) != text(penalized),
      "a strong repetition penalty should alter at least one word choice once the same " +
        "sentence genuinely repeats verbatim within one turn")
  }

  "noRepeatNgramSize" should "reach real ASR generation without erroring" taggedAs SlowTest in {
    val baseline = run(freshDiarizerAsrDefault(), repeatedSpeech)
    val constrained =
      run(freshDiarizerAsrDefault().setNoRepeatNgramSize(2), repeatedSpeech)

    println(s"[noRepeatNgramSize] baseline=${text(baseline)} | constrained=${text(constrained)}")
    assert(text(baseline).nonEmpty)
    assert(text(constrained).nonEmpty)
  }

  "randomSeed" should "make sampled generation reproducible across separate calls with the same seed" taggedAs SlowTest in {
    val run1 = run(
      freshDiarizerAsrDefault().setDoSample(true).setTemperature(0.9).setRandomSeed(2024L),
      audio("synthetic_2speaker"))
    val run2 = run(
      freshDiarizerAsrDefault().setDoSample(true).setTemperature(0.9).setRandomSeed(2024L),
      audio("synthetic_2speaker"))
    val run3 = run(
      freshDiarizerAsrDefault().setDoSample(true).setTemperature(0.9).setRandomSeed(99L),
      audio("synthetic_2speaker"))

    println(s"[randomSeed] seed=2024 run1=${text(run1)}")
    println(s"[randomSeed] seed=2024 run2=${text(run2)}")
    println(s"[randomSeed] seed=99   run3=${text(run3)}")
    assert(
      text(run1) == text(run2),
      "the same randomSeed must reproduce byte-identical sampled output across separate calls")
  }

  "beamSize/nReturnSequences" should "be accepted but have no effect on the bundled Whisper model's output" taggedAs SlowTest in {
    val default = run(freshDiarizerAsrDefault(), audio("synthetic_2speaker"))

    val withBeamAndReturnSeqs = run(
      freshDiarizerAsrDefault().setBeamSize(4).setNReturnSequences(3),
      audio("synthetic_2speaker"))

    println(s"[beamSize/nReturnSequences no-op] default=${text(default)}")
    println(
      s"[beamSize/nReturnSequences no-op] beamSize=4,nReturnSequences=3=${text(withBeamAndReturnSeqs)}")
    assert(text(default) == text(withBeamAndReturnSeqs))
    assert(
      withBeamAndReturnSeqs.length == default.length,
      "nReturnSequences=3 must still produce exactly one transcript per turn, not three, since " +
        "the beam-search path it would otherwise control is never used")
  }

  private def freshDiarizerAsrDefault(): SpeakerDiarizer =
    freshDiarizerWithAsr().setMinDurationOn(0.3f)

  private lazy val audioAssembler: AudioAssembler = new AudioAssembler()
    .setInputCol("audio_content")
    .setOutputCol("audio_assembler")

  private def freshPipelineDiarizer(): SpeakerDiarizer =
    freshDiarizer().setMinDurationOn(0.3f)

  "SpeakerDiarizer" should "produce correct results through a real AudioAssembler -> Pipeline -> transform(df) flow" taggedAs SlowTest in {
    import spark.implicits._
    val diarizer = freshPipelineDiarizer()
    val pipeline = new Pipeline().setStages(Array(audioAssembler, diarizer))

    val df = Seq(audio("synthetic_2speaker")).toDF("audio_content")
    val transformed = pipeline.fit(df).transform(df)

    val perRow = Annotation.collect(transformed, "speakers")
    assert(perRow.length == 1, "expected exactly one output row for one input row")
    val annotations = perRow.head.toSeq
    assert(annotations.nonEmpty, "expected at least one detected turn")

    val pipelineSpeakers = annotations.map(_.metadata("speaker")).distinct
    println(s"[pipeline] speakers=$pipelineSpeakers turns=${annotations.length}")
    assert(
      pipelineSpeakers.length >= 2,
      s"expected >=2 distinct speakers from synthetic_2speaker.wav via the real pipeline, got $pipelineSpeakers")

    val direct = run(freshPipelineDiarizer(), audio("synthetic_2speaker"))
    assert(
      annotations.length == direct.length,
      s"pipeline path found ${annotations.length} turns, direct batchAnnotate found ${direct.length}")
    assert(
      annotations.map(_.metadata("speaker")).distinct.length == speakers(direct).distinct.length)
  }

  it should "process every row of a multi-row DataFrame independently and correctly" taggedAs SlowTest in {
    import spark.implicits._
    val diarizer = freshPipelineDiarizer()
    val pipeline = new Pipeline().setStages(Array(audioAssembler, diarizer))

    val df = Seq(audio("single_speaker"), audio("synthetic_2speaker")).toDF("audio_content")
    val transformed = pipeline.fit(df).transform(df)

    val perRow = Annotation.collect(transformed, "speakers")
    assert(perRow.length == 2, s"expected 2 output rows, got ${perRow.length}")

    val row0Speakers = perRow(0).map(_.metadata("speaker")).distinct
    val row1Speakers = perRow(1).map(_.metadata("speaker")).distinct
    println(s"[pipeline multi-row] row0=${row0Speakers.toSeq} row1=${row1Speakers.toSeq}")
    assert(
      row0Speakers.length == 1,
      s"row 0 (single_speaker.wav) should show exactly 1 speaker, got ${row0Speakers.toSeq}")
    assert(
      row1Speakers.length >= 2,
      s"row 1 (synthetic_2speaker.wav) should show >=2 speakers, got ${row1Speakers.toSeq}")
  }

  it should "produce an output column with the documented SPEAKER annotation schema" taggedAs SlowTest in {
    import spark.implicits._
    val diarizer = freshPipelineDiarizer()
    val pipeline = new Pipeline().setStages(Array(audioAssembler, diarizer))

    val df = Seq(audio("single_speaker")).toDF("audio_content")
    val transformed = pipeline.fit(df).transform(df)

    val speakersField = transformed.schema("speakers")
    val elementFields =
      speakersField.dataType
        .asInstanceOf[org.apache.spark.sql.types.ArrayType]
        .elementType
        .asInstanceOf[org.apache.spark.sql.types.StructType]
        .fieldNames
        .toSet
    println(s"[pipeline schema] speakers element fields=$elementFields")
    assert(
      Set("annotatorType", "begin", "end", "result", "metadata", "embeddings")
        .subsetOf(elementFields),
      s"expected the standard Annotation struct fields, got $elementFields")
  }
}

trait SpeakerDiarizerTestFixtures { this: AnyFlatSpec =>

  protected val scratch: String = sys.env.getOrElse(
    "SPARKNLP_SPEAKER_DIARIZER_TEST_SCRATCH",
    "/private/tmp/claude-501/-Users-abdullah-Documents-spark-nlp--claude-worktrees-opt-125m-research-90713f/61bcb16c-f914-49ff-af4a-c56c4c5e7ac6/scratchpad")
  protected val pkgPath: String = s"$scratch/models/spark_nlp_pkg"
  protected val whisperPkgPath: String = s"$scratch/models/whisper_pkg"
  protected val td: String = s"$scratch/models/testdata"
  protected val spark = ResourceHelper.spark

  protected def loadFloats(path: String): Array[Float] = {
    val src = Source.fromFile(path)
    try {
      src.getLines().map(_.trim.toFloat).toArray
    } finally src.close()
  }

  protected def audio(name: String): Array[Float] = loadFloats(s"$td/${name}_floats.txt")

  protected def freshDiarizer(): SpeakerDiarizer =
    SpeakerDiarizer
      .loadSavedModel(pkgPath, spark)
      .setInputCols("audio_assembler")
      .setOutputCol("speakers")
      .setTranscribe(false)
      .setSamplingRate(16000)

  protected def freshDiarizerWithAsr(): SpeakerDiarizer =
    SpeakerDiarizer
      .loadSavedModel(pkgPath, spark, asrModelPath = Some(whisperPkgPath))
      .setInputCols("audio_assembler")
      .setOutputCol("speakers")
      .setSamplingRate(16000)

  protected def run(diarizer: SpeakerDiarizer, samples: Array[Float]): Seq[Annotation] = {
    val row = Array(AnnotationAudio(AnnotatorType.AUDIO, samples, Map.empty))
    diarizer.batchAnnotate(Seq(row)).head
  }

  protected def speakers(anns: Seq[Annotation]): Seq[String] = anns.map(_.metadata("speaker"))
  protected def confidences(anns: Seq[Annotation]): Seq[Double] =
    anns.map(_.metadata("confidence").toDouble)
  protected def text(anns: Seq[Annotation]): String = anns.map(_.result).mkString(" ")
}
