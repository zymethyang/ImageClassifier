function [ categoryClassifier ] = train(dataset_dir)
imgSets = imageSet(dataset_dir,'recursive');
[trainingSets, testSets] = partition(imgSets, 0.3, 'randomize');
bag = bagOfFeatures(trainingSets,'Verbose',true)
categoryClassifier = trainImageCategoryClassifier(trainingSets, bag)
confMatrix = evaluate(categoryClassifier, testSets)
mean(diag(confMatrix))
end

