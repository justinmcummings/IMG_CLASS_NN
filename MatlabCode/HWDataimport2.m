%MNIST
% data files from http://cis.jhu.edu/~sachin/digit/digit.html
clear
close all

fid=fopen('data7','r'); % open the file corresponding to digit 7

i=0;
n=1000;
for j=1:n
    [t,N]=fread(fid,[28 28],'uint8'); % read in the first training example and store it in a 28x28 size matrix t1
    t=t';
    imagesc(t)
    colormap(gray)
    axis square
    pause(.001)
    i=i+1;
    X7(i,:)=reshape(t,1,28*28);  % store each 7 as 1 x 28^2 row in X
end


fid=fopen('data6','r'); % open the file corresponding to digit 7

i=0;
n=1000;
for j=1:n
    [t,N]=fread(fid,[28 28],'uint8'); % read in the first training example and store it in a 28x28 size matrix t1
    t=t';
    imagesc(t)
    colormap(gray)
    axis square
    pause(.001)
    i=i+1;
    X6(i,:)=reshape(t,1,28*28);  % store each 7 as 1 x 28^2 row in X
end

Xtest = [X6(101:end,:);X7(101:end,:)]';
ytest = [ones(1,900),zeros(1,900)];
X = [X6(1:100,:);X7(1:100,:)]';
y = [ones(1,100),zeros(1,100)];