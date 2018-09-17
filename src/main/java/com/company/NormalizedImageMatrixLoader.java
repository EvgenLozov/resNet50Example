package com.company;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;

public class NormalizedImageMatrixLoader {

    private NativeImageLoader imageLoader;
    private DataNormalization scaler;

    public NormalizedImageMatrixLoader(NativeImageLoader imageLoader, DataNormalization scaler) {
        this.imageLoader = imageLoader;
        this.scaler = scaler;
    }

    public INDArray load(InputStream inputStream){

        try {
            INDArray indArray =  imageLoader.asMatrix(inputStream);
            scaler.transform(indArray);

            indArray.putColumn(0, indArray.getColumn(0).add(-0.485).mul(1.0/0.229));
            indArray.putColumn(1, indArray.getColumn(1).add(-0.456).mul(1.0/0.224));
            indArray.putColumn(2, indArray.getColumn(2).add(-0.406).mul(1.0/0.225));

            return indArray;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
