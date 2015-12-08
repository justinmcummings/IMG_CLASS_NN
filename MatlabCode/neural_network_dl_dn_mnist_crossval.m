%% Help Function
% This should be the main function for the neural network
%% Read in 6 and 7 of the MNIST Data Set
clear all
clc
[six,seven] = mnist_code(1000);
tstart= tic;
for ll = 1:5
    for kk = 1:5
        indx = randperm(100,75);
        X = vertcat(six(indx,:), seven(indx,:));
        [sx1,sx2] = size(X);
        m = sx1;
        for i = 1:sx2
            X(:,i) = X(:,i) - (mean(X(:,i)));
        end
        
        indx2 = setdiff(1:100,indx);
        X_val = vertcat(six(indx2,:),seven(indx2,:));
        [sx1_val,sx2_val] = size(X_val);
        for i = 1:sx2_val
            X_val(:,i) = X_val(:,i) - (mean(X_val(:,i)));
        end
        
        indx3 = setdiff(1:1000,indx);
        indx3 = setdiff(indx3,indx2);
        X_hol = vertcat(six(indx3,:),seven(indx3,:));
        [sx1_hol,sx2_hol] = size(X_hol);
        for i = 1:sx2_hol
            X_hol(:,i) = X_hol(:,i) - (mean(X_hol(:,i)));
        end
        % Label the data set
        y = [-1*ones(1,m/2),ones(1,m/2)]';
        y_val = [-1*ones(1,sx1_val/2),ones(1,sx1_val/2)]';
        y_hol = [-1*ones(1,sx1_hol/2),ones(1,sx1_hol/2)]';
        
        % layers in the neural network
        nl = 3;
        
        %% Feed-Forward
        l2_extran = 0;
        w{1} = rand(sx2,sx2+l2_extran)./10000;
        w{2} = rand(sx2+l2_extran,1)./10000;
        
        b{1} = rand(sx2+l2_extran,1)./10000;
        b{2} = rand/10000;
        
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
        etano = 5*10^-3;
        count = 1;
        while err > 2.5*10^-1
            eta = etano/(1+sqrt(count)*.005);
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
                for j = 1:sx1
                    % Do the feed-forward algorithm for the current layer
                    z{l+1}(:,j) = w{l}'*a{l}(:,j)+b{l};
                    a{l+1}(:,j) = sigmoid(z{l+1}(:,j));
                end
            end
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
        
        a_val{1} = X_val';
        for l = 1:nl-1
            for j = 1:sx1_val
                % Do the feed-forward algorithm for the current layer
                z_val{l+1}(:,j) = w{l}'*a_val{l}(:,j)+b{l};
                a_val{l+1}(:,j) = sigmoid(z_val{l+1}(:,j));
            end
        end
        err_rate(kk) = sum(sign(a_val{3}') ~= y_val)/length(y_val);
        w1_store{kk} = w{1};
        w2_store{kk} = w{2};
        b1_store{kk} = b{1};
        b2_store{kk} = b{2};
    end
    [~,minerr] = min(err_rate);
    w_best{1} = w1_store{minerr};
    w_best{2} = w2_store{minerr};
    b_best{1} = b1_store{minerr};
    b_best{2} = b2_store{minerr};
    
    a_hol{1} = X_hol';
    for l = 1:nl-1
        for j = 1:sx1_hol
            % Do the feed-forward algorithm for the current layer
            z_hol{l+1}(:,j) = w_best{l}'*a_hol{l}(:,j)+b_best{l};
            a_hol{l+1}(:,j) = sigmoid(z_hol{l+1}(:,j));
        end
    end
    err_perform(ll) = sum(sign(a_hol{3}') ~= y_hol)/length(y_hol);
end
tstop = toc(tstart)/60;