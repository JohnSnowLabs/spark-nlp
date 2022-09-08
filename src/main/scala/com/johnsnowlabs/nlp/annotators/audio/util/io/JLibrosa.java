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

package com.johnsnowlabs.nlp.annotators.audio.util.io;

import java.io.*;
import java.math.RoundingMode;
import java.nio.Buffer;
import java.text.DecimalFormat;
import java.util.ArrayList;


class FileFormatNotSupportedException extends Exception {
    public FileFormatNotSupportedException(String message) {
        super(message);
    }
}

class WavFileException extends Exception {
    private static final long serialVersionUID = 1L;

    public WavFileException(String message) {
        super(message);
    }
}


public class JLibrosa {
    private int BUFFER_SIZE = 4096;
    private int noOfFrames = -1;
    private int sampleRate = -1;
    private int noOfChannels = -1;
    private double fMax = 22050.0;
    public int getNoOfChannels() {
        return this.noOfChannels;
    }
    public void setNoOfChannels(int noOfChannels) {
        this.noOfChannels = noOfChannels;
    }
    public JLibrosa() {
    }
    public int getNoOfFrames() {
        return this.noOfFrames;
    }
    public void setNoOfFrames(int noOfFrames) {
        this.noOfFrames = noOfFrames;
    }
    public void setSampleRate(int sampleRate) {
        this.sampleRate = sampleRate;
        this.fMax = (double)sampleRate / 2.0;
    }


    /**
     * This function is used to load the audio file and read its Numeric Magnitude
     * Feature Values.
     *
     * @param
     * @param sampleRate
     * @param readDurationInSeconds
     * @return
     * @throws IOException
     * @throws WavFileException
     * @throws FileFormatNotSupportedException
     */
    private float[][] readMagnitudeValuesFromFile(BufferedInputStream stream, int sampleRate, int readDurationInSeconds, int offsetDuration)
            throws IOException, WavFileException, FileFormatNotSupportedException {



        com.johnsnowlabs.nlp.annotators.audio.util.io.WavFile wavFile = null;

        wavFile = com.johnsnowlabs.nlp.annotators.audio.util.io.WavFile.openWavFile(stream);
        int mNumFrames = (int) (wavFile.getNumFrames());
        int mSampleRate = (int) wavFile.getSampleRate();
        int mChannels = wavFile.getNumChannels();

        int totalNoOfFrames = mNumFrames;
        int frameOffset = offsetDuration * mSampleRate;
        int tobeReadFrames = readDurationInSeconds * mSampleRate;

        if(tobeReadFrames > (totalNoOfFrames - frameOffset)) {
            tobeReadFrames = totalNoOfFrames - frameOffset;
        }

        if (readDurationInSeconds != -1) {
            mNumFrames = tobeReadFrames;
            wavFile.setNumFrames(mNumFrames);
        }


        this.setNoOfChannels(mChannels);
        this.setNoOfFrames(mNumFrames);
        this.setSampleRate(mSampleRate);


        if (sampleRate != -1) {
            mSampleRate = sampleRate;
        }

        // Read the magnitude values across both the channels and save them as part of
        // multi-dimensional array

        float[][] buffer = new float[mChannels][mNumFrames];
        long readFrameCount = 0;
        //for (int i = 0; i < loopCounter; i++) {
        readFrameCount = wavFile.readFrames(buffer, mNumFrames, frameOffset);
        //}

        if(wavFile != null) {
            wavFile.close();
        }

        return buffer;

    }


    /**
     * This function loads the audio file, reads its Numeric Magnitude Feature
     * values and then takes the mean of amplitude values across all the channels and
     * convert the signal to mono mode by taking the average. This method reads the audio file
     * post the mentioned offset duration in seconds.
     *
     * @param stream
     * @param sampleRate
     * @param readDurationInSeconds
     * @param offsetDuration
     * @return
     * @throws IOException
     * @throws WavFileException
     * @throws FileFormatNotSupportedException
     */
    public float[] loadAndReadWithOffset(BufferedInputStream stream, int sampleRate, int readDurationInSeconds, int offsetDuration)
            throws IOException, WavFileException, FileFormatNotSupportedException {
        float[][] magValueArray = readMagnitudeValuesFromFile(stream, sampleRate, readDurationInSeconds, offsetDuration);

        DecimalFormat df = new DecimalFormat("#.#####");
        df.setRoundingMode(RoundingMode.CEILING);

        int mNumFrames = this.getNoOfFrames();
        int mChannels = this.getNoOfChannels();

        // take the mean of amplitude values across all the channels and convert the
        // signal to mono mode

        float[] meanBuffer = new float[mNumFrames];


        for (int q = 0; q < mNumFrames; q++) {
            double frameVal = 0;
            for (int p = 0; p < mChannels; p++) {
                frameVal = frameVal + magValueArray[p][q];
            }
            meanBuffer[q] = Float.parseFloat(df.format(frameVal / mChannels));
        }

        return meanBuffer;

    }

    /**
     * This function loads the audio file, reads its Numeric Magnitude Feature
     * values and then takes the mean of amplitude values across all the channels and
     * convert the signal to mono mode
     *
     * @param stream
     * @param sampleRate
     * @param readDurationInSeconds
     * @return
     * @throws IOException
     * @throws WavFileException
     * @throws FileFormatNotSupportedException
     */
    public ArrayList<Float> loadAndReadAsListWithOffset(BufferedInputStream stream, int sampleRate, int readDurationInSeconds, int offsetDuration)
            throws IOException, WavFileException, FileFormatNotSupportedException {

        float[][] magValueArray = readMagnitudeValuesFromFile(stream, sampleRate, readDurationInSeconds, offsetDuration);

        DecimalFormat df = new DecimalFormat("#.#####");
        df.setRoundingMode(RoundingMode.CEILING);

        int mNumFrames = this.getNoOfFrames();
        int mChannels = this.getNoOfChannels();

        // take the mean of amplitude values across all the channels and convert the
        // signal to mono mode
        float[] meanBuffer = new float[mNumFrames];
        ArrayList<Float> meanBufferList = new ArrayList<Float>();
        for (int q = 0; q < mNumFrames; q++) {
            double frameVal = 0;
            for (int p = 0; p < mChannels; p++) {
                frameVal = frameVal + magValueArray[p][q];
            }
            meanBufferList.add(Float.parseFloat(df.format(frameVal / mChannels)));

        }

        return meanBufferList;

    }

    /**
     * This function loads the audio file, reads its Numeric Magnitude Feature
     * values and then takes the mean of amplitude values across all the channels and
     * convert the signal to mono mode
     *
     * @param stream
     * @param sampleRate
     * @param readDurationInSeconds
     * @return
     * @throws IOException
     * @throws WavFileException
     * @throws FileFormatNotSupportedException
     */
    public ArrayList<Float> loadAndReadAsList(BufferedInputStream stream, int sampleRate, int readDurationInSeconds)
            throws IOException, WavFileException, FileFormatNotSupportedException {

        ArrayList<Float> meanBufferList = loadAndReadAsListWithOffset(stream, sampleRate, readDurationInSeconds, 0);
        return meanBufferList;

    }
}