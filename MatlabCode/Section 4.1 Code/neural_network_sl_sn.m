%% Help Function
% This should be the main function for the neural network

%% Initialization
clear
clc
%% Generate two distributions

agen = 4.1;
bgen = 6;
x1 = agen + (bgen-agen).*rand(50,2);

agen = 3;
bgen = 4;
x2 = agen + (bgen-agen).*rand(50,2);

%% Initialization
% layers in the neural network
nl = 3;

% input
X = vertcat(x1,x2);
X(:,1) = X(:,1) - (mean(X(:,1)));
X(:,2) = X(:,2) - (mean(X(:,2)));
[sx1,sx2] = size(X);
X = [X,ones(sx1,1)];

[sx1,sx2] = size(X);
% output
y = [ones(1,length(x1)),-1*ones(1,length(x2))]';

m = sx1;

plot(X(:,1),X(:,2),'o')
axis tight

%% Feed-Forward
w = rand(sx2,1)./1000;
for j = 1:m
    % Do the feed-forward algorithm for the current layer
    z(:,j) = w'*X(j,:)';
    y_hat(j,:) = sigmoid(z(:,j));
end

%% Back-Propagation    
err = inf;
eta = 1;
count = 1;
while err > 1.5*10^-1
    grad = zeros(size(w));
    for j = 1:m % use for GD
        delta = 2*(y_hat(j)-y(j))*grad_sigmoid(z(j));    
        grad = grad + delta.*X(j,:)';  
    end % use for GD
    
    w = w - eta*(1/(2*m))*grad;
    for j = 1:m
        % Do the feed-forward algorithm for the current layer
        z(:,j) = w'*X(j,:)';
        y_hat(j,:) = sigmoid(z(:,j));
    end
    
    err_old = err;
    err = norm(y_hat-y)
    
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
agen = 4.1;
bgen = 6;
x1 = agen + (bgen-agen).*rand(100,2);

agen = 3;
bgen = 4;
x2 = agen + (bgen-agen).*rand(100,2);

X = vertcat(x1,x2);
X(:,1) = X(:,1) - (mean(X(:,1)));
X(:,2) = X(:,2) - (mean(X(:,2)));
X = [X,ones(length(x1)+length(x2),1)];

y = [ones((length(x1)+length(x2))/2,1); -ones((length(x1)+length(x2))/2,1)];

a{1} = X';
for j = 1:(length(x1)+length(x2))
    % Do the feed-forward algorithm for the current layer
    z(:,j) = w'*X(j,:)';
    y_hat(j,:) = sigmoid(z(:,j));
end
%% Comparison to RLS
figure(1)
plot(X(:,1),X(:,2),'o')
hold on
lambda = 10^-5;
plot(linspace(-2,2,100),-(linspace(-2,2,100)*w(1)+w(3))/(w(2)))

w_rls = pinv(X'*X + lambda*eye(size(X'*X)))*X'*y;
plot(linspace(-2,2,100),-(linspace(-2,2,100)*w_rls(1)+w_rls(3))/(w_rls(2)))
legend('Data','Neural Net','Regularized Least Squares')
hold off

numerr_nn = sum(sign(y_hat) ~= y)
numerr_rls = sum(sign(X*w_rls) ~= y)

err_rate_nn = sum(sign(y_hat) ~= y)/length(y)
err_rate_rls = sum(sign(X*w_rls) ~= y)/length(y)

%% Boundary Plot
[x1,x2] = meshgrid(min(X(:,1)):.1:max(X(:,1)),min(X(:,2)):.1:max(X(:,2)));
x1 = reshape(x1,length(x1)^2,1);
x2 = reshape(x2,length(x2)^2,1);

X_b = [x1,x2];
X_b = [X_b,ones(length(x1),1)];

for j = 1:length(x1)
    % Do the feed-forward algorithm for the current layer
    z(:,j) = w'*X_b(j,:)';
    y_hat(j,:) = sigmoid(z(:,j));
end
y_hat = sign(y_hat);
indx = find(y_hat == -1);

figure(2)
plot(X_b(indx,1),X_b(indx,2),'go')
hold on
indx2 = find(y_hat == 1);
plot(X_b(indx2,1),X_b(indx2,2),'ro')
hold on
plot(X(:,1),X(:,2),'o')