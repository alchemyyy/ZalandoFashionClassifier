function averageError = getNetworkAccuracy(net, trainImages, trainLabels, startIndex, endIndex)
    averageError = 0;
        for index = startIndex:endIndex
            %adjust for matlab stupid base 1 crap
            if (trainLabels(index) ~= (net.getNumericalOutput(trainImages(:,index))))
                averageError = averageError + 1;
            end
        end
        averageError = averageError / (endIndex - startIndex);
end

