//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package org.tensorflow.ndarray.buffer;

import java.io.Serializable;

public interface DataBuffer<T> extends Serializable {
    long size();

    boolean isReadOnly();

    T getObject(long var1);

    DataBuffer<T> setObject(T var1, long var2);

    default DataBuffer<T> read(T[] dst) {
        return this.read(dst, 0, dst.length);
    }

    DataBuffer<T> read(T[] var1, int var2, int var3);

    default DataBuffer<T> write(T[] src) {
        return this.write(src, 0, src.length);
    }

    DataBuffer<T> write(T[] var1, int var2, int var3);

    DataBuffer<T> copyTo(DataBuffer<T> var1, long var2);

    default DataBuffer<T> offset(long index) {
        return this.slice(index, this.size() - index);
    }

    default DataBuffer<T> narrow(long size) {
        return this.slice(0L, size);
    }

    DataBuffer<T> slice(long var1, long var3);

    default DataBufferWindow<? extends DataBuffer<T>> window(long size) {
        throw new UnsupportedOperationException();
    }

    default <R> R accept(DataStorageVisitor<R> visitor) {
        return visitor.fallback();
    }

    boolean equals(Object var1);
}
