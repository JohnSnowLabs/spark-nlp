//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package org.tensorflow.ndarray.impl.buffer.raw;

import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

final class UnsafeMemoryHandle implements Serializable {
    final Object object;
    final long baseOffset;
    long byteOffset;
    final long byteSize;
    final long scale;
    final long size;

    static UnsafeMemoryHandle fromArray(Object array, int length) {
        return fromArray(array, 0, length);
    }

    static UnsafeMemoryHandle fromArray(Object array, int arrayOffset, int length) {
        long scale = (long)UnsafeReference.UNSAFE.arrayIndexScale(array.getClass());
        int baseOffset = UnsafeReference.UNSAFE.arrayBaseOffset(array.getClass());
        return new UnsafeMemoryHandle(array, (long)baseOffset + (long)arrayOffset * scale, (long)length * scale, scale);
    }

    static UnsafeMemoryHandle fromAddress(long address, long byteSize, long scale) {
        return new UnsafeMemoryHandle(address, byteSize, scale);
    }

    long size() {
        return this.size;
    }

    byte getByte(long index) {
        return UnsafeReference.UNSAFE.getByte(this.object, this.align(index));
    }

    void setByte(byte value, long index) {
        UnsafeReference.UNSAFE.putByte(this.object, this.align(index), value);
    }

    boolean getBoolean(long index) {
        return UnsafeReference.UNSAFE.getBoolean(this.object, this.align(index));
    }

    void setBoolean(boolean value, long index) {
        UnsafeReference.UNSAFE.putBoolean(this.object, this.align(index), value);
    }

    short getShort(long index) {
        return UnsafeReference.UNSAFE.getShort(this.object, this.align(index));
    }

    void setShort(short value, long index) {
        UnsafeReference.UNSAFE.putShort(this.object, this.align(index), value);
    }

    int getInt(long index) {
        return UnsafeReference.UNSAFE.getInt(this.object, this.align(index));
    }

    void setInt(int value, long index) {
        UnsafeReference.UNSAFE.putInt(this.object, this.align(index), value);
    }

    float getFloat(long index) {
        return UnsafeReference.UNSAFE.getFloat(this.object, this.align(index));
    }

    void setFloat(float value, long index) {
        UnsafeReference.UNSAFE.putFloat(this.object, this.align(index), value);
    }

    double getDouble(long index) {
        return UnsafeReference.UNSAFE.getDouble(this.object, this.align(index));
    }

    void setDouble(double value, long index) {
        UnsafeReference.UNSAFE.putDouble(this.object, this.align(index), value);
    }

    long getLong(long index) {
        return UnsafeReference.UNSAFE.getLong(this.object, this.align(index));
    }

    void setLong(long value, long index) {
        UnsafeReference.UNSAFE.putLong(this.object, this.align(index), value);
    }

    void copyTo(UnsafeMemoryHandle memory, long length) {
        UnsafeReference.UNSAFE.copyMemory(this.object, this.byteOffset, memory.object, memory.byteOffset, length * this.scale);
    }

    UnsafeMemoryHandle offset(long index) {
        long offset = this.scale(index);
        return new UnsafeMemoryHandle(this.object, this.byteOffset + offset, this.byteSize - offset, this.scale);
    }

    UnsafeMemoryHandle narrow(long size) {
        return new UnsafeMemoryHandle(this.object, this.byteOffset, this.scale(size), this.scale);
    }

    UnsafeMemoryHandle slice(long index, long size) {
        return new UnsafeMemoryHandle(this.object, this.byteOffset + this.scale(index), this.scale(size), this.scale);
    }

    UnsafeMemoryHandle rescale(long scale) {
        if (this.object != null) {
            throw new IllegalStateException("Raw heap memory cannot be rescaled");
        } else {
            return new UnsafeMemoryHandle((Object)null, this.byteOffset, this.byteSize, scale);
        }
    }

    void rebase(long index) {
        this.byteOffset = this.baseOffset + this.scale(index);
    }

    boolean isArray() {
        return this.object != null;
    }

    <A> A array() {
        return (A) this.object;
    }

    int arrayOffset(Class<?> arrayClass) {
        return (int)((this.byteOffset - (long)UnsafeReference.UNSAFE.arrayBaseOffset(arrayClass)) / this.scale);
    }

    ByteBuffer toArrayByteBuffer() {
        return ByteBuffer.wrap((byte[])((byte[])this.object), (int)this.byteOffset - UnsafeReference.UNSAFE.arrayBaseOffset(byte[].class), (int)this.size);
    }

    ShortBuffer toArrayShortBuffer() {
        return ShortBuffer.wrap((short[])((short[])this.object), (int)((this.byteOffset - (long)UnsafeReference.UNSAFE.arrayBaseOffset(short[].class)) / this.scale), (int)this.size);
    }

    IntBuffer toArrayIntBuffer() {
        return IntBuffer.wrap((int[])((int[])this.object), (int)((this.byteOffset - (long)UnsafeReference.UNSAFE.arrayBaseOffset(int[].class)) / this.scale), (int)this.size);
    }

    LongBuffer toArrayLongBuffer() {
        return LongBuffer.wrap((long[])((long[])this.object), (int)((this.byteOffset - (long)UnsafeReference.UNSAFE.arrayBaseOffset(long[].class)) / this.scale), (int)this.size);
    }

    FloatBuffer toArrayFloatBuffer() {
        return FloatBuffer.wrap((float[])((float[])this.object), (int)((this.byteOffset - (long)UnsafeReference.UNSAFE.arrayBaseOffset(float[].class)) / this.scale), (int)this.size);
    }

    DoubleBuffer toArrayDoubleBuffer() {
        return DoubleBuffer.wrap((double[])((double[])this.object), (int)((this.byteOffset - (long)UnsafeReference.UNSAFE.arrayBaseOffset(double[].class)) / this.scale), (int)this.size);
    }

    private UnsafeMemoryHandle(Object object, long baseOffset, long byteSize, long scale) {
        this.object = object;
        this.baseOffset = baseOffset;
        this.byteOffset = baseOffset;
        this.byteSize = byteSize;
        this.scale = scale;
        this.size = byteSize / scale;
    }

    private UnsafeMemoryHandle(long address, long byteSize, long scale) {
        this((Object)null, address, byteSize, scale);
    }

    private long align(long index) {
        return this.byteOffset + index * this.scale;
    }

    private long scale(long value) {
        return value * this.scale;
    }
}
