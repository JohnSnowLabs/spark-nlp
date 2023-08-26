package com.johnsnowlabs.nlp.annotators.audio.feature_extractor

import breeze.linalg.{DenseMatrix, DenseVector, csvread}
import breeze.signal.support.WindowFunctions.hanningWindow
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.TestUtils.tolerantDoubleEq
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File

class AudioUtilsTest extends AnyFlatSpec {

  behavior of "AudioUtilsTest"

  it should "hertzToMel and melToHertz" taggedAs FastTest in {
    val hz: Seq[Double] = (-44000 to 44000 by 2000).map(_.toDouble)
    val expectedMels: Seq[Double] = Seq(-660.0, -630.0, -600.0, -570.0, -540.0, -510.0, -480.0,
      -450.0, -420.0, -390.0, -360.0, -330.0, -300.0, -270.0, -240.0, -210.0, -180.0, -150.0,
      -120.0, -90.0, -60.0, -30.0, 0.0, 25.081880157308323, 35.163760314616646, 41.06128214340673,
      45.245640471924965, 48.491280943849944, 51.14316230071505, 53.38529604052273,
      55.32752062923329, 57.040684129505124, 58.573161101158256, 59.95945514881271,
      61.225042458023374, 62.389269924950604, 63.46717619783106, 64.47068292994834,
      65.40940078654161, 66.29119067024521, 67.12256428681346, 67.90897626477928,
      68.65504125846658, 69.36469802662114, 70.04133530612103)

    hz.zip(expectedMels).foreach { case (freq, expectedMel) =>
      val mel = AudioUtils.hertzToMel(freq)
      assert(mel == expectedMel)
      assert(AudioUtils.melToHertz(expectedMel) === freq)
    }
  }

  it should "melFilterBank" taggedAs FastTest in {
    val maxFrequency = 8000.0
    val minFrequency = 0.0
    val numFrequencyBins = 201
    val numMelFilters = 80
    val samlingRate = 16000

    val expectedMelFilters = csvread(new File("src/test/resources/audio/csv/mel_filters.csv"))

    val melFilterBank = AudioUtils.melFilterBank(
      numFrequencyBins,
      numMelFilters,
      minFrequency,
      maxFrequency,
      samlingRate)

    melFilterBank.valuesIterator.zip(expectedMelFilters.valuesIterator).foreach {
      case (filterVal, expectedVal) =>
        assert(filterVal === expectedVal)
    }
  }

  it should "padReflective" taggedAs FastTest in {
    val a: DenseVector[Double] = DenseVector(1.0d, 2.0d, 3.0d, 4.0d, 5.0d)

    val expected = DenseVector(3, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0)

    assert(AudioUtils.padReflective(a, (2, 2)) == expected)
  }

}
