function categoricalLabels = convertLabelsToCategorical(location, integerLabels)
load(fullfile(location,'batches.meta.mat'));
categoricalLabels = categorical(integerLabels, 0:9, label_names);
end