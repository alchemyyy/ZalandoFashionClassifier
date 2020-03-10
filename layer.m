classdef layer < matlab.mixin.Copyable
    properties
        p%input
        w%weight
        b%bias
        n%sum
        a%output
        s%sensitivity
        f%activation function copy. pass the actual function
        fd%derivative of activation function
        filter%The image kernel for this layer
    end
    methods
        function newLayer = layer(inputVectorLength, outputVectorLength, activationFunctionCopy, activationFunctionDerivativeCopy, weightBiasInitDistType, weightBiasInitMin, weightBiasInitMax, filter)
            if (strcmp(weightBiasInitDistType,'normal'))
                weightBiasInitRange = (weightBiasInitMax - weightBiasInitMin) / 3;%empirical rule
                newLayer.w = normrnd(0, weightBiasInitRange, outputVectorLength, inputVectorLength) + weightBiasInitMin;
                newLayer.b = normrnd(0, weightBiasInitRange, outputVectorLength, 1) + weightBiasInitMin;
            else
                weightBiasInitRange = (weightBiasInitMax - weightBiasInitMin);
                newLayer.w = weightBiasInitRange * rand(weightBiasInitRange, outputVectorLength, inputVectorLength) + weightBiasInitMin;
                newLayer.b = weightBiasInitRange * rand(weightBiasInitRange, outputVectorLength, 1) + weightBiasInitMin;
            end
            newLayer.f = activationFunctionCopy;
            newLayer.fd = activationFunctionDerivativeCopy;
            newLayer.filter = filter;
        end
        
        function feedForward(obj, p)
            obj.p = conv2(p, obj.filter, 'same');
            obj.n = (obj.w * p + obj.b);
            obj.a = obj.f(obj.n);
        end
        
        function applySens(obj, alpha)
            as = alpha * obj.s;
            obj.w = obj.w - as * obj.p';
            obj.b = obj.b - as;
        end
    end
end