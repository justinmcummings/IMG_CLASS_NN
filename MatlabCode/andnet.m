ed function h = andnet(X);
X = X(:);
X = [X;1];
W = [20,20,-30]';

out = neuron(X,W);

h = round(out);

end
