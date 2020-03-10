function submissionGenerator()
net = evalin('base','bestnetwork');
testImages = evalin('base','testImages');
testIDs = evalin('base','testIDs');


testSetLabels = zeros(length(testImages),1); 
for i = 1:length(testImages)
    t = net.getNumericalOutput(testImages(:,i));
    testSetLabels(i,:) = t(1);
end

%ADD HEADER MANUALLY

csvwrite('output_addheadertomeplease.csv', [testIDs testSetLabels]);
disp("DONE");