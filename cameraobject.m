load ('webcamsSceneReconstruction.mat');
data = load('fasterRCNNVehicleTrainingData.mat');
trainingData = data.vehicleTrainingData;
trainingData.imageFilename = fullfile(toolboxdir('vision'),'visiondata', trainingData.imageFilename);
%forming layers
layers = data.layers
options = trainingOptions('sgdm','InitialLearnRate', 1e-6, ...
    'MaxEpochs', 1, ...
    'CheckpointPath', tempdir);
% Training detector. 
detector = trainFasterRCNNObjectDetector(trainingData, layers, options)
% Testing the Fast R-CNN detector on a test image.
webcamlist
a=webcam(2);
framecount=1;
[height width channel]=size(snapshot(a));
while framecount < 5
    img = snapshot(a);
    im_Left = img(:, 1 : width/2, :);
    %figure;
    %b=imshow(im_Left);
    im_Right = img(:, width/2 +1: width, :);
% Run detector.
[bbox, score, label] = detect(detector, im_Left);
[bbox, score, label]= detect(detector,im_Right);
detectedImg1 = insertShape(im_Left, 'Rectangle', bbox);
detectedImg2= insertShape(im_Right, 'Rectangle',bbox);
figure
imshow(detectedImg1)
figure
imshow(detectedImg2)
leftimage=undistortImage(im_Left,stereoParams.CameraParameters1);
rightimage=undistortImage(im_Right,stereoParams.CameraParameters2);
center1 = detectedImg1(1:2) + detectedImg1(3:4)/0.005;
center2 = detectedImg2(1:2) + detectedImg2(3:4)/0.005;
point3d = triangulate(center1, center2, stereoParams);
distanceInMeters = norm(point3d)/1000;
distanceAsString = sprintf('%0.2f meters', distanceInMeters);
leftimage = insertObjectAnnotation(leftimage,'rectangle',center1,distanceAsString);
rightimage = insertObjectAnnotation(rightimage,'rectangle',center2, distanceAsString);
%leftimage = insertShape(detectedImg1,'Rectangle',bboxes,'LineWidth',3);
%rightimage = insertShape(detectedImg2,'Rectangle',bboxes,'LineWidth',3);
leftimage = insertShape(leftimage,'FilledRectangle',detectedImg1);
rightimage = insertShape(rightimage,'FilledRectangle',detectedImg2);
imshowpair(leftimage, rightimage, 'montage');
framecount=framecount+1;
end