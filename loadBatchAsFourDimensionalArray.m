function [XBatch, TBatch] = loadBatchAsFourDimensionalArray(location, batchFileName)
load(fullfile(location,batchFileName));
XBatch = data';
XBatch = reshape(XBatch, 32,32,3,[]);
XBatch = permute(XBatch, [2 1 3 4]);
TBatch = convertLabelsToCategorical(location, labels);
end