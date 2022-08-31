package com.johnsnowlabs.nlp.audio;

/**
 * Custom Exception Class to handle errors occurring while loading/reading Wav file.
 * 
 * @author abhi-rawat1
 *
 */
public class WavFileException extends Exception {

	private static final long serialVersionUID = 1L;

	public WavFileException(final String message) {
        super(message);
    }
	
}
