package com.johnsnowlabs.nlp.annotators.parser.typdep.io;

import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyInstance;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public abstract class DependencyReader {

    BufferedReader reader;
    boolean isLabeled=true;

    public static DependencyReader createDependencyReader() {
        return new Conll09Reader();
    }

    public abstract DependencyInstance nextInstance() throws IOException;

    public void startReading(String file) throws IOException {
        reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8));
    }

    public void close() throws IOException { if (reader != null) reader.close(); }

}
