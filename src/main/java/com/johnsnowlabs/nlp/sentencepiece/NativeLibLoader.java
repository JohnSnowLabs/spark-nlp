package com.johnsnowlabs.nlp.sentencepiece;

import java.io.*;

/**
 * @author https://github.com/levyfan
 * @see https://github.com/levyfan/sentencepiece-jni
 */
class NativeLibLoader {
    private static final String SENTENCEPIECE_LIB = "sentencepiece_jni";
    private static final String MAPPED_NAME = System.mapLibraryName(SENTENCEPIECE_LIB);
    private static final String SENTENCEPIECE_PATH_PROPERTY = "sentencepiece.lib.path";

    /**
     * Create a temp file that copies the resource from current JAR archive
     * <p/>
     * The file from JAR is copied into system temp file.
     * The temporary file is deleted after exiting.
     * Method uses String as filename because the pathname is "abstract", not system-dependent.
     * <p/>
     * The restrictions of {@link File#createTempFile(String, String)} apply to
     * {@code path}.
     * @param path Path to the resources in the jar
     * @return The created temp file.
     * @throws IOException When the temp file could not be created
     * @throws IllegalArgumentException When the file name contains invalid letters
     * @author https://github.com/levyfan
     */
    static String createTempFileFromResource(String path) throws IOException, IllegalArgumentException {
        // Obtain filename from path
        if (!path.startsWith("/")) {
            throw new IllegalArgumentException("The path has to be absolute (start with '/').");
        }

        String[] parts = path.split("/");
        String filename = (parts.length > 1) ? parts[parts.length - 1] : null;

        // Split filename to prexif and suffix (extension)
        String prefix = "";
        String suffix = null;
        if (filename != null) {
            parts = filename.split("\\.", 2);
            prefix = parts[0];
            suffix = (parts.length > 1) ? "." + parts[parts.length - 1] : null; // Thanks, davs! :-)
        }

        // Check if the filename is okay
        if (filename == null || prefix.length() < 3) {
            throw new IllegalArgumentException("The filename has to be at least 3 characters long.");
        }
        // Prepare temporary file
        File temp = File.createTempFile(prefix, suffix);
        temp.deleteOnExit();

        if (!temp.exists()) {
            throw new FileNotFoundException("File " + temp.getAbsolutePath() + " does not exist.");
        }

        // Prepare buffer for data copying
        byte[] buffer = new byte[1024];
        int readBytes;

        // Open and check input stream
        InputStream is = NativeLibLoader.class.getResourceAsStream(path);
        if (is == null) {
            throw new FileNotFoundException("File " + path + " was not found inside JAR.");
        }

        // Open output stream and copy data between source file in JAR and the temporary file
        OutputStream os = new FileOutputStream(temp);
        try {
            while ((readBytes = is.read(buffer)) != -1) {
                os.write(buffer, 0, readBytes);
            }
        } finally {
            // If read/write fails, close streams safely before throwing an exception
            os.close();
            is.close();
        }
        return temp.getAbsolutePath();
    }

    /**
     * This method tries to find the location of the sentencepiece-jni library.
     * @return a path for System.loadLibrary to use
     * @throws IOException
     * @author https://github.com/alexander-n-thomas
     */
    static String getPathToNativeSentencePieceLib() throws IOException {
        // path passed in explicitly
        if (System.getProperty(SENTENCEPIECE_PATH_PROPERTY) != null) {
            return System.getProperty(SENTENCEPIECE_PATH_PROPERTY);
        // file is on the classpath
        } else if (NativeLibLoader.class.getResourceAsStream("/lib/" + MAPPED_NAME) != null) {
            return createTempFileFromResource("/lib/" + MAPPED_NAME);
        } else {
            return null;
        }
    }
}

