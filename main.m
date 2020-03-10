function [] = main()
    %graph generation
    mseFig = figure(2);
    msePlot = plot([0], [0], '-');
    hold on
    trainAccPlot = plot([0], [0], '-');
    testAccPlot = plot([0], [0], '-');
    hold off
    title(strcat('Figure ', num2str(mseFig.Number), ': Mean Error$$^2$$ per Epoch for MNIST'), 'Interpreter', 'latex');
    xlabel('Epoch');
    ylabel('MSE');
    tic
    
    %[trainImages, trainLabels, testImages, testLabels] = readMNIST();
    trainImages = evalin('base', 'trainImages');
    trainLabels = evalin('base', 'trainLabels');
    trainLabelsActual = evalin('base', 'trainLabelsActual');
    trainImages = trainImages / 255;
    trainImages = SVDPreprocessor(trainImages, 10);
    %image(255 * trainImages(:,:,:,10));
    %disp(double(trainLabels(10)));

    %training parameters
    trainingSetLength = 2000;
    %testSetLength = 1000;
    numberOfEpochs = 100000;
    
    %learning rate parameters
    learnRate = 0.02;
    VLRMSETriggerPercentage = 0.08;
    VLRDecreasePercentage = 0.25;
    VLRIncreasePercentage = 0.005;

    %define filters here.
    filterGaussian = (1/16)*[1 2 1; 2 4 2; 1 2 1];
    blankFilter = [1 1 1; 1 1 1; 1 1 1];
    randFilter1 = rand(3);
    randFilter4 = rand(3);
    edgeFilter = [-1 -1 -1; -1 8 -1; -1 -1 -1];
    %network parameters
    %layer(inputVectorLength, outputVectorLength, activationFunctionCopy, activationFunctionDerivativeCopy, weightBiasInitDistType, weightBiasInitMin, weightBiasInitMax)      
    layerHidden1 = layer(784, 225, @afRelu, @afdRelu, 'normal', 0, .35, filterGaussian);
    layerHidden2 = layer(225, 100, @afRelu, @afdRelu, 'normal', 0, .35, randFilter4);
    layerOutput = layer(100, 10, @logsig, @afdLogsig, 'normal', 0, .3, filterGaussian);
    net = network(learnRate, [layerHidden1 layerHidden2 layerOutput]);
    MSE_old = 100;
    lowestError = 100;
    
    %load saved network. IF NOT LOADED OLD NETWORK IS LOST
    %net = evalin('base','bestnetwork');
    %lowestError = evalin('base','lowestError');
    %MSE_old = evalin('base','lowestErrorMSE');
    %end load saved network
    %because matlab is stupid, we can't use a for loop. because if we need to backtrack our epoch (for VLR) it won't let us.
    epoch = 1;
    while epoch < numberOfEpochs
        MSE_new = 0;
        net_old = copy(net);
            permVec = randperm(length(trainLabels));
    shuffledTrainImages = zeros(size(trainImages));
    shuffledTrainLabels = zeros(size(trainLabels));
    shuffledTrainLabelsActual = zeros(size(trainLabelsActual));
    for i=1:length(trainLabels)
        shuffledTrainImages(:,permVec(i)) = trainImages(:,i); 
        shuffledTrainLabels(permVec(i),:) = trainLabels(i,:);
        shuffledTrainLabelsActual(permVec(i)) = trainLabelsActual(i);
        
    end
    trainImages = shuffledTrainImages;
    trainLabels = shuffledTrainLabels;
    trainLabelsActual = shuffledTrainLabelsActual;
        
        %begin epoch
        for index = 1:trainingSetLength
             MSE_new = MSE_new + power(net.train(shuffledTrainImages(:, index), shuffledTrainLabels(index,:)),2);
        end
        %end epoch

        %test all data that was just trained
        %getNetworkAccuracy(net, trainImages, startIndex, endIndex)
        TrainingError = getNetworkAccuracy(net, trainImages, trainLabelsActual, 1, trainingSetLength);
        TestError = getNetworkAccuracy(net, trainImages, trainLabelsActual, trainingSetLength, length(trainImages));
        %end test all data
        
        MSE_new = MSE_new / trainingSetLength;
        %adjust learning rate
        MSE_delta = MSE_new - MSE_old;
        if (MSE_delta > 0)
            if (MSE_delta > VLRMSETriggerPercentage)
                disp("MSE Jumped! Adjusting Learning Rate..");
                %revert network state and lower learning rate
                net = net_old;
                net.learnRate = net.learnRate * (1 - VLRDecreasePercentage);
                %treat this epoch as failed. revert everything, do not increment epoch
                continue;
            end
        else
            net.learnRate = net.learnRate * (1 + VLRIncreasePercentage);
        end
        MSE_old = MSE_new;
        %end adjust learning rate
        
        %save best results
        if (TestError < lowestError)
            assignin('base','bestnetwork',net);
            assignin('base','lowestError',TestError);
            assignin('base','lowestErrorMSE',MSE_new);
            lowestError = TestError;
        end
        %end save best results
        
        %display results of this epoch
        disp("Epoch:" + epoch + "  MSE:" + MSE_new + "  MSEDelta:" + MSE_delta + " Test Accuracy:" + (1 - TestError) + "  learnRate:" + net.learnRate + "  best:" + (1 - lowestError));
        %disp(temp_MSE);
        msePlot.XData(epoch) = epoch;
        msePlot.YData(epoch) = MSE_new;
        trainAccPlot.XData(epoch) = epoch;
        trainAccPlot.YData(epoch) = TrainingError;
        testAccPlot.XData(epoch) = epoch;
        testAccPlot.YData(epoch) = TestError;
        if (toc > 0.2)
            drawnow
            tic
        end
        
        %ITERATE EPOCH MANUALLY
        epoch = epoch + 1;
    end
end
