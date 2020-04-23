%%
load sss.mat
%%
size(trainingImages)
%%
numImageCategories = 10;
categories(trainingLabels)
%%
figure
thumbnails = trainingImages(:,:,:,1:100);
montage(thumbnails)
%%
% 
[height,width,numChannels, ~] = size(trainingImages);

imageSize = [height width numChannels];
inputLayer = imageInputLayer(imageSize)
%%
% Convolutional layer parameters
filterSize = [5 5];
numFilters = 32;

middleLayers = [
    
% The first convolutional layer has a bank of 32 5x5x3 filters.
convolution2dLayer(filterSize,numFilters,'Padding',2)


% Next add the ReLU layer:
reluLayer()


maxPooling2dLayer(3,'Stride',2)

% Repeat the 3 core layers to complete the middle of the network.
convolution2dLayer(filterSize,numFilters,'Padding',2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

convolution2dLayer(filterSize,2 * numFilters,'Padding',2)
reluLayer()
maxPooling2dLayer(3,'Stride',2)

]
%%
finalLayers = [
    
fullyConnectedLayer(64)

% Add an ReLU non-linearity.
reluLayer

fullyConnectedLayer(numImageCategories)


softmaxLayer
classificationLayer
]
%%
layers = [
    inputLayer
    middleLayers
    finalLayers
    ]
%%
% Set the network training options
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'Verbose', true);
%%

doTraining = false;

if doTraining    
    % Train a network.
    cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);
else
   
    load('rcnnStopSigns.mat','cifar10Net')       
end
%%
% Extract the first convolutional layer weights
w = cifar10Net.Layers(2).Weights;

% rescale the weights to the range [0, 1] for better visualization
w = rescale(w);

figure
montage(w)
%%
% Run the network on the test set.
[YTest,score] = classify(cifar10Net, testImages);

% Calculate the accuracy.
accuracy = sum(YTest == testLabels)/numel(testLabels)
%%
[m,order] = confusionmat(testLabels,YTest);
figure
cm = confusionchart(m,order);
%%
for i=1:10000
    k=max(score(i,:));
    score1(i)=k;
end
%%
a=unique(testLabels);
for i=1:numel(a)
   p=string(a(i));
[X,Y] = perfcurve(testLabels,score1,p);
subplot(2,5,i)
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title(sprintf('ROC for %s',p))
end
%%
% Load the ground truth data
data = load('stopSignsAndCars.mat', 'stopSignsAndCars');
stopSignsAndCars = data.stopSignsAndCars;


visiondata = fullfile(toolboxdir('vision'),'visiondata');
stopSignsAndCars.imageFilename = fullfile(visiondata, stopSignsAndCars.imageFilename);

summary(stopSignsAndCars)
%%

stopSigns = stopSignsAndCars(:, {'imageFilename','stopSign'});

% Display one training image and the ground truth bounding boxes
I = imread(stopSigns.imageFilename{3});
I = insertObjectAnnotation(I,'Rectangle',stopSigns.stopSign{1},'stop sign','LineWidth',8);

figure
imshow(I)
%%

doTraining = false;

if doTraining
    
    % Set training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 100, ...
        'Verbose', true);
    
    % Train an R-CNN object detector. This will take several minutes.    
    rcnn = trainRCNNObjectDetector(stopSigns, cifar10Net, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])
else
    
    load('rcnnStopSigns.mat','rcnn')       
end
%%
%%
% Read test image
testImage = imread('image035.jpg');
imshow(testImage)
% Detect stop signs
[bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128)
%%
% Display the detection results
[score, idx] = max(score)

bbox = bboxes(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation);

figure
imshow(outputImage)
%%

