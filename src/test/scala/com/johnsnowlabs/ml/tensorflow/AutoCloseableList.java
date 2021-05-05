package com.johnsnowlabs.ml.tensorflow;

import java.util.ArrayList;
import java.util.Collection;

public final class AutoCloseableList<E extends AutoCloseable> extends ArrayList<E>
        implements AutoCloseable {

    public AutoCloseableList(Collection<? extends E> c) {
        super(c);
    }

    @Override
    public void close() {
        Exception toThrow = null;
        for (AutoCloseable c : this) {
            try {
                c.close();
            } catch (Exception e) {
                toThrow = e;
            }
        }
        if (toThrow != null) {
            throw new RuntimeException(toThrow);
        }
    }
}
