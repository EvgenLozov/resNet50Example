package com.company;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.function.Function;

public class NormalizedImageMatrixLoaderProvider {

    public Function<String,INDArray> byFile(){
        NativeImageLoader imageLoader = new NativeImageLoader(224  , 224, 3);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        NormalizedImageMatrixLoader loader = new NormalizedImageMatrixLoader(imageLoader,scaler);

        return fileName -> {
            try(InputStream is = new FileInputStream(fileName)) {

                return loader.load(is);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        };
    }
}
