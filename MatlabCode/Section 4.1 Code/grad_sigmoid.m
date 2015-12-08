function df_sigmoid = grad_sigmoid(z)
    f_sigmoid = sigmoid(z);
    %df_sigmoid = f_sigmoid.*(1-f_sigmoid); % 0 1 function
    df_sigmoid = (1-(f_sigmoid).^2); % -1 1 function
end