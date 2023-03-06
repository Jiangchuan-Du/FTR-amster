function f1 = reparam_traj(f)
%REPARAM_TRAJ Reparameterize the trajectory according to the Euclidean
%distance.
% Input:
%   f - N by B matrix where each column is the trajectory of a biomarker.
%
% Output:
%   f1 - reparameterized trajectory

% x = x(:); y = y(:);

[N,B] = size(f);  

% make it a circular list since we are dealing with closed contour
% x = [x;x(1)];
% y = [y;y(1)];

% dx = x([2:N+1])- x(1:N);
% dy = y([2:N+1])- y(1:N);
% d = sqrt(dx.*dx+dy.*dy);  % compute the distance from previous node for point 2:N+1

df = f(2:end,:) - f(1:end-1,:);
d = sqrt(sum(df.^2, 2));

d = [0;d];   % point 1 to point 1 is 0 

% now compute the arc length of all the points to point 1
% we use matrix multiply to achieve summing 
% M = length(d);
% d = (d'*uppertri(M,M))';

d = cumsum(d);

% now ready to reparametrize the closed curve in terms of arc length
maxd = d(end);

% if (maxd/RES<3)
%    error('RES too big compare to the length of original curve');
% end

di = linspace(0, maxd, N);

f1 = zeros(N,B);
for i = 1:B
    [~, idx, ~] = unique(d);
    d = d(idx);
    f = f(idx,:);
    f1(:,i) = interp1(d, f(:,i), di);
end
    
% xi = interp1(d,x,di');
% yi = interp1(d,y,di');

% N = length(xi);

% if (maxd - di(length(di)) <RES/2)  % deal with end boundary condition
%    xi = xi(1:N-1);
%    yi = yi(1:N-1);
% end

end

