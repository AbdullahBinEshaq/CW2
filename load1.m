% Return CIFAR-10 Training and Test data.
function [XTrain, TTrain, XTest, TTest] = load1(dataLocation)         
            
            location = fullfile(dataLocation, 'cifar-10-batches-mat');
            
            [XTrain1, TTrain1] = loadBatchAsFourDimensionalArray(location, 'data_batch_1.mat');
            [XTrain2, TTrain2] = loadBatchAsFourDimensionalArray(location, 'data_batch_2.mat');
            [XTrain3, TTrain3] = loadBatchAsFourDimensionalArray(location, 'data_batch_3.mat');
            [XTrain4, TTrain4] = loadBatchAsFourDimensionalArray(location, 'data_batch_4.mat');
            [XTrain5, TTrain5] = loadBatchAsFourDimensionalArray(location, 'data_batch_5.mat');
            
            XTrain = cat(4, XTrain1, XTrain2, XTrain3, XTrain4, XTrain5);
            TTrain = [TTrain1; TTrain2; TTrain3; TTrain4; TTrain5];
            
            [XTest, TTest] = loadBatchAsFourDimensionalArray(location, 'test_batch.mat');
                      
end