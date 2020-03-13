function imageData = SVDPreprocessor(imageData, featureVectorLength)
    for i = 1:length(imageData)
        tempMat = reshape(imageData(:, i), [28, 28]);
        [U, S, V] = svd(tempMat, 'econ');
        approx = U(:, 1:featureVectorLength) * S(1:featureVectorLength, 1:featureVectorLength) * V(:, 1:featureVectorLength)';
        approx = reshape(approx.',1,[]);
        tempMat = conv2(tempP, matGaussian, 'same');
        imageData(:, i) = approx;
    end
end
