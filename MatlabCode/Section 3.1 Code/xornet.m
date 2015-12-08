function h = xornet(X)

X = X(:);
X = [X;1];

W11 = []';
a1 = neuron(X,W11);

W12 = []';
a2 = neuron(X,W12);

x2 = [a1 a2 1]';
W21 = []';
out = neuron(x2,W21);

h =round(out);
end
