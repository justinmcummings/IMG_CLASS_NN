%% Help Function
% This should be the main function for the neural network
%% Generate two distributions
clear all
clc

agen = 3;
bgen = 4;
x1 = agen + (bgen-agen).*rand(50,2);

agen = 4.1;
bgen = 5;
x2 = agen + (bgen-agen).*rand(50,2);

agen = 5.1;
bgen = 6;
x3 = agen + (bgen-agen).*rand(50,2);
%% Initialization
% layers in the neural network
nl = 3;

% input
X = vertcat(x1,x2,x3);
X(:,1) = X(:,1) - (mean(X(:,1)));
X(:,2) = X(:,2) - (mean(X(:,2)));
[sx1,sx2] = size(X);

% output
y = [-ones(1,length(x1)),1*ones(1,length(x2)),-ones(1,length(x1))]';

m = sx1;
plot(X(1:length(x1),1),X(1:length(x1),2),'go')
hold on
plot(X((length(x1)+1):(length(x1)+length(x2)),1),X((length(x1)+1):(length(x1)+length(x2)),2),'ro')
hold on
plot(X((length(x1)+length(x2)+1):m,1),X((length(x1)+length(x2)+1):m,2),'go')
axis tight

%% Feed-Forward
l2_extran = 0;
w{1} = rand(sx2,sx2+l2_extran)./1;
w{2} = rand(sx2+l2_extran,1)./1;

b{1} = rand(sx2+l2_extran,1)./1;
b{2} = rand/1;

a{1} = X';
for l = 1:nl-1
    for j = 1:m
        % Do the feed-forward algorithm for the current layer
        z{l+1}(:,j) = w{l}'*a{l}(:,j)+b{l};
        a{l+1}(:,j) = sigmoid(z{l+1}(:,j));
    end
end

%% Back-Propagation
err = inf;
etano = 5*10^-1;
count = 1;
while err > 5*10^-1
    eta = etano/(1+sqrt(count)*.01);
    grad_2_weights = zeros(size(w{2}));
    grad_2_bias = zeros(size(b{2}));
    grad_1_weights = zeros(size(w{1}));
    grad_1_bias = zeros(size(b{1}));
    for j = 1:m % use for GD
        delta = 2*(a{3}(j)-y(j))*grad_sigmoid(z{3}(j));
        grad_2_weights = grad_2_weights + (delta*a{2}(:,j));
        grad_2_bias =  grad_2_bias + delta;
        for k = 1:(sx2+l2_extran)
            delta_bar(:,k) = delta*w{2}(k,:)*grad_sigmoid(z{2}(k,j));
            grad_1_bias(k) =  grad_1_bias(k) + delta_bar(k);
            grad_1_weights(:,k) = grad_1_weights(:,k) + delta_bar(:,k).*X(j,:)';
        end
    end % use for GD
    w{2} = w{2} - eta*(1/(2*m))*grad_2_weights;
    b{2} = b{2} - eta*(1/(2*m))*grad_2_bias;
    
    w{1} = w{1} - eta*(1/(2*m))*grad_1_weights;
    b{1} = b{1} - eta*(1/(2*m))*grad_1_bias;
    
    err_old = err;
    % Feed-Forward
    a{1} = X';
    for l = 1:nl-1
        for j = 1:m
            % Do the feed-forward algorithm for the current layer
            z{l+1}(:,j) = w{l}'*a{l}(:,j)+b{l};
            a{l+1}(:,j) = sigmoid(z{l+1}(:,j));
        end
    end
    % Check the Error
    err = norm(a{3}'-y)
    
    if norm(err_old-err) < 10^-10
        fprintf('Error threshold not achievable, minimum solution found.\n')
        break;
    elseif(err > err_old)
        fprintf('Divergence has occured after %1.0f iterations.\n',count)
        break;
    end
    count = count + 1;
end

%% validation
agen = 3;
bgen = 4;
x1 = agen + (bgen-agen).*rand(50,2);

agen = 4.1;
bgen = 5;
x2 = agen + (bgen-agen).*rand(50,2);

agen = 5.1;
bgen = 6;
x3 = agen + (bgen-agen).*rand(50,2);

y_val =[-ones(length(x1),1);ones(length(x2),1);-ones(length(x3),1)];

X = vertcat(x1,x2,x3);
X(:,1) = X(:,1) - (mean(X(:,1)));
X(:,2) = X(:,2) - (mean(X(:,2)));

a_val{1} = X';
for l = 1:nl-1
    for j = 1:(length(x1)+length(x2)+length(x3))
        % Do the feed-forward algorithm for the current layer
        z_val{l+1}(:,j) = w{l}'*a_val{l}(:,j)+b{l};
        a_val{l+1}(:,j) = sigmoid(z_val{l+1}(:,j));
    end
end
err_val = sum(sign(a_val{3}') ~= y_val)
%% Boundary Plot
[x1,x2] = meshgrid(min(X(:,1)):.1:max(X(:,1)),min(X(:,2)):.1:max(X(:,2)));
[wx1,lx1] = size(x1);
x1 = reshape(x1,wx1*lx1,1);
x2 = reshape(x2,wx1*lx1,1);

X_b = [x1,x2];
a_b{1} = X_b';
for l = 1:nl-1
    for j = 1:length(x1)
        % Do the feed-forward algorithm for the current layer
        z_b{l+1}(:,j) = w{l}'*a_b{l}(:,j)+b{l};
        a_b{l+1}(:,j) = sigmoid(z_b{l+1}(:,j));
    end
end

y_hat = sign(a_b{3}');
figure(2)
indx = find(y_hat == -1);
plot(X_b(indx,1),X_b(indx,2),'go')
hold on
indx2 = find(y_hat == 1);
plot(X_b(indx2,1),X_b(indx2,2),'ro')
hold on
plot(X(:,1),X(:,2),'x')
axis tight