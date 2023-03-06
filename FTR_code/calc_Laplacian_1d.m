function L = calc_Laplacian_1d(delta_ts)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Ti = length(delta_ts) + 1;

W = zeros(Ti,Ti);
for i = 1:Ti
    if i == 1
        W(i,2) = convert_delta_to_w(delta_ts(i));
    elseif i == Ti
        W(i,Ti-1) = convert_delta_to_w(delta_ts(i-1));
    else
        W(i,i-1) = convert_delta_to_w(delta_ts(i-1));
        W(i,i+1) = convert_delta_to_w(delta_ts(i));
    end
end

D = diag(sum(W,2));

L = D - W;

end

function w = convert_delta_to_w(delta)
w = 1/delta;
end