classdef network < matlab.mixin.Copyable
    properties
        layers
        layersLength
        learnRate
    end
    
    methods
        function obj = network(newLearnRate, layers)
            obj.learnRate = newLearnRate;
            obj.layers = layers;
            obj.layersLength = length(obj.layers);
        end
        
        function mse = train(obj, trainValue, trainLabel)
            %get target error and calculate mse
            %normally
            performanceIndex = double(trainLabel)' - obj.feedForwardNetwork(trainValue);
            %cross entropy with softmax
            %targetError = -sum(double(trainLabel)' .* log(obj.feedForwardNetwork(trainValue)));
            mse = sum(power(performanceIndex, 2));
            
            %calculate sensitivities (gradients or whatever)
            %normally
            obj.layers(obj.layersLength).s = -2 * performanceIndex .* obj.layers(obj.layersLength).f(obj.layers(obj.layersLength).a);
            %cross entropy with softmax
            %obj.layers(obj.layersLength).s = -2 * targetError' .* (double(trainLabel)' - obj.feedForwardNetwork(trainValue));
            
            for layerIndex = obj.layersLength - 1:-1:1
                obj.layers(layerIndex).s = obj.layers(layerIndex).fd((obj.layers(layerIndex).a)) .* obj.layers(layerIndex + 1).w' * obj.layers(layerIndex + 1).s;
            end
            
            %apply them
            obj.applySensNetwork();
        end
        
        function performanceIndex = trainSoftmax(obj, trainValue, trainLabel)
            %get target error and calculate mse
            performanceIndex = -sum(double(trainLabel)' .* log(obj.feedForwardNetwork(trainValue)));
            %calculate sensitivities (gradients or whatever)
            obj.layers(obj.layersLength).s = -2 * performanceIndex' .* (double(trainLabel)' - obj.feedForwardNetwork(trainValue));
            for layerIndex = obj.layersLength - 1:-1:1
                obj.layers(layerIndex).s = obj.layers(layerIndex).fd((obj.layers(layerIndex).a)) .* obj.layers(layerIndex + 1).w' * obj.layers(layerIndex + 1).s;
            end
            %apply them
            obj.applySensNetwork();
        end

        function applySensNetwork(obj)
            for layerIndex = obj.layersLength:-1:1
                obj.layers(layerIndex).applySens(obj.learnRate);
            end
        end
        
        function output = feedForwardNetwork(obj, value)
            obj.layers(1).feedForward(value);
            for layerIndex = 2:obj.layersLength
                obj.layers(layerIndex).feedForward(obj.layers(layerIndex - 1).a);
            end
            output = obj.layers(obj.layersLength).a;
        end
        
        function output = getNumericalOutput(obj, value)
            netout = obj.feedForwardNetwork(value);
            maxim = find(netout == max(netout));
            %convert to base 0
            output = maxim - 1;
        end
    end
end