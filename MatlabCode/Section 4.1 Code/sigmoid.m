%% Help Function
% z is calculated in the feed-forward step
function f_sigmoid = sigmoid(z)
    %f_sigmoid = 1./(1+exp(-z)); % 0 1 function
    f_sigmoid = (exp(z)-exp(-z))./(exp(z)+exp(-z)); % -1 1 function
end