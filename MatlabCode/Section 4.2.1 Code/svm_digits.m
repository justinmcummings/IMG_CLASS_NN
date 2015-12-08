%% Classification of Handwritten Digits Using Support Vector Machines
% By Akhil Sundararajan

clear;
close all;
clc;
%% Read in Digit Data (Sixes and Sevens)
% data files from http://cis.jhu.edu/~sachin/digit/digit.html

fid=fopen('data7','r'); % open the file corresponding to digit 7

i=0;
n=1000;
for j=1:n
    [t,N]=fread(fid,[28 28],'uint8'); % read in the first training example and store it in a 28x28 size matrix t1
    t=t';
    imagesc(t)
    colormap(gray)
    axis square
    pause(.01)
    i=i+1;
    X7(i,:)=reshape(t,1,28*28);  % store each 7 as 1 x 28^2 row in X
end

fid=fopen('data6','r'); % open the file corresponding to digit 6

figure;

i=0;
n=1000;
for j=1:n
    [t,N]=fread(fid,[28 28],'uint8'); % read in the first training example and store it in a 28x28 size matrix t1
    t=t';
    imagesc(t)
    colormap(gray)
    axis square
    pause(.01)
    i=i+1;
    X6(i,:)=reshape(t,1,28*28);  % store each 6 as 1 x 28^2 row in X
end
%% Training an SVM Classifier
% Assign label of 1 if digit is a 7 and -1 if a 6.

A = [X6; X7];
b = [ones(1000,1); -1*ones(1000,1)];

training_sizes = 100;
% training_sizes = [1 5 10 20 30 50 100 200 400 500 900];

avg_err = zeros(1,length(training_sizes));
avg_err_GK = zeros(1,length(training_sizes));
avg_err_PK = zeros(1,length(training_sizes));
avg_err_MLP = zeros(1,length(training_sizes));

turnOnLK = 0;
turnOnGK = 0;
turnOnPK = 0;
turnOnMLP = 1;
showPlot = 0;

for ii = 1:length(training_sizes)
N_tr = training_sizes(ii);

iter = 100;
err_rate = zeros(1,iter);
err_rate_GK = zeros(1,iter);
err_rate_PK = zeros(1,iter);
err_rate_MLP = zeros(1,iter);

for kk = 1:iter

    train_ind = [randperm(1000,N_tr),1000+randperm(1000,N_tr)];
    test_ind = setdiff(1:2000,train_ind);

A_train = A(train_ind,:);
b_train = b(train_ind);
A_test = A(test_ind,:);
b_test = b(test_ind);

if (turnOnLK)
    % Train SVM Classifier (Linear Kernel)
    svmStruct = svmtrain(???,???);
    pred = svmclassify(???,???);

    num_mistakes = 0;
    for i = 1:length(pred)
       if (pred(i) ~= b_test(i))
           num_mistakes = num_mistakes+1;
       end
    end

    err_rate(kk) = num_mistakes/length(b_test);
end


if (turnOnGK)
    % Training a Kernelized SVM Classifier (Gaussian Kernel)
    svmStruct = svmtrain(A_train,b_train,'kernel_function','rbf');
    pred_GK = svmclassify(svmStruct,A_test);

    num_mistakes_GK = 0;
    for i = 1:length(pred_GK)
       if (pred_GK(i) ~= b_test(i))
           num_mistakes_GK = num_mistakes_GK+1;
       end
    end

    err_rate_GK(kk) = num_mistakes_GK/length(b_test);
end

if (turnOnPK)
    % Training a Kernelized SVM Classifier (Polynomial Kernel)
    svmStruct = svmtrain(A_train,b_train,'kernel_function','polynomial');
    pred_PK = svmclassify(svmStruct,A_test);

    num_mistakes_PK = 0;
    for i = 1:length(pred_PK)
       if (pred_PK(i) ~= b_test(i))
           num_mistakes_PK = num_mistakes_PK+1;
       end
    end
    err_rate_PK(kk) = num_mistakes_PK/length(b_test);   
end

if (turnOnMLP)
    % Training a Kernelized SVM Classifier (MLP Kernel)
    svmStruct = svmtrain(A_train,b_train,'kernel_function','mlp','mlp_params',[0.1,-0.5]);
    pred_MLP = svmclassify(svmStruct,A_test);

    num_mistakes_MLP = 0;
    for i = 1:length(pred_MLP)
       if (pred_MLP(i) ~= b_test(i))
           num_mistakes_MLP = num_mistakes_MLP+1;
       end
    end
    err_rate_MLP(kk) = num_mistakes_MLP/length(b_test);
end

end

avg_err(ii) = mean(err_rate);
avg_err_GK(ii) = mean(err_rate_GK);
avg_err_PK(ii) = mean(err_rate_PK);
avg_err_MLP(ii) = mean(err_rate_MLP);
end

if (showPlot)
    figure;
    hold on;
    plot(training_sizes,avg_err)
    plot(training_sizes,avg_err_GK)
    plot(training_sizes,avg_err_PK)
    plot(training_sizes,avg_err_MLP)
    hold off;
    xlabel('Training Size');
    ylabel('Error Rate');
    title('Error Rate vs. Training Size: SVM Classifiers');
    legend('Linear Kernel','Gaussian Kernel','Cubic Kernel','MLP Kernel','location','best');
end