function h = ornet(X)
X = X(:);
X = [X;1];
W = []';

out = neuron(X,W);

h = round(out);

end
