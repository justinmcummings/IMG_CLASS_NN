function h = neuron(X,W,TYPE)
if nargin < 3
    TYPE = 1;
end

X = X(:);
W = W(:);

z = W'*X;

if TYPE == 1
    % gradient = sig(z)*(1-sig(z))
    h = 1./(1+exp(-1*z));
end
if TYPE == 2
    % graddient = 1 - sig(z)^2;
    h = (exp(z) - exp(-z))./(exp(z) + exp(-z));
end

end
