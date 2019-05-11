function [  ] = use(categoryClassifier,img_name)
img = imread(img_name);
imshow(img)
[labelIdx, score] = predict(categoryClassifier, img);
categoryClassifier.Labels(labelIdx)
end

