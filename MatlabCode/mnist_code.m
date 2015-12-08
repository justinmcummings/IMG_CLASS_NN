function [six,seven] = mnist_code(n)
%%
%MNIST
% data files from http://cis.jhu.edu/~sachin/digit/digit.html
fid=fopen('data7.txt','r'); % open the file corresponding to digit 7

i=0;
for j=1:n
    [t,N]=fread(fid,[28 28],'uint8'); % read in the first training example and store it in a 28x28 size matrix t1
    t=t';
    i=i+1;
    seven(i,:)=reshape(t,1,28*28);  % store each 7 as 1 x 28^2 row in X
end
fclose(fid);

fid=fopen('data6.txt','r'); % open the file corresponding to digit 6

i=0;
for j=1:n
    [t,N]=fread(fid,[28 28],'uint8'); % read in the first training example and store it in a 28x28 size matrix t1
    t=t';
    i=i+1;
    six(i,:)=reshape(t,1,28*28);  % store each 6 as 1 x 28^2 row in X
end
fclose(fid);
end