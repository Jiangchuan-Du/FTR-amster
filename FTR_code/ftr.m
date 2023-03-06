function [s,f] = ftr(X, xmin, xmax, rec_pts, mu, sigma, nu, ifplot)
[F_y,F_x] = ecdf(X);

if sigma == 0
    % take the inverse
    s = linspace(0, 1, rec_pts);
    f = interp1(F_y, F_x, s);
    return;
end
F_x = F_x(2:end);
F_y = F_y(2:end);

fontsize = 40;

%[xs, s1] = deconvolution_with_known_minmax(F_x, F_y, xmin, xmax, sigma);
[xs, s1] = deconvolution_without_known_minmax(F_x, F_y, mu, sigma, nu, ifplot);
s = linspace(0, 1, rec_pts);
f = interp1(s1, xs, s);
if ifplot == 1
    figure;
    %plot(F_x,F_y,'black','LineWidth',10)
    plot(F_x,F_y,'black')
    set(gca,'fontweight','bold')
    %set(gca,'fontweight','bold','FontSize',70,'LineWidth',10)
    %axis tight;
    %set(gca,'xticklabel',[],'yticklabel',[])
    box off;
    %title('\boldmath$F_j(x)$','interpreter','latex','FontSize',70)
    title('\boldmath$F_j(x)$','interpreter','latex')
    saveas(figure,'F:\Research\Project\1_Filtered_trajectory_recovery\Fig\CDF.png');
    figure;
    %plot(f,s,'black','LineWidth',10)
    plot(f,s,'black')
    set(gca,'fontweight','bold')
    %set(gca,'fontweight','bold','FontSize',70,'LineWidth',10)
    %axis tight;
    %set(gca,'xticklabel',[],'yticklabel',[])
    box off;
    %title('\boldmath$H\circ f_j^{-1}$','interpreter','latex','FontSize',70)
    title('\boldmath$H\circ f_j^{-1}$','interpreter','latex')
    saveas(figure,'F:\Research\Project\1_Filtered_trajectory_recovery\Fig\deconvoluted.png');
    figure;
    %plot(s,f,'black','LineWidth',10)
    plot(f,s,'black')
    set(gca,'fontweight','bold')
    %set(gca,'fontweight','bold','FontSize',70,'LineWidth',10)
    %axis tight;
    %set(gca,'xticklabel',[],'yticklabel',[])
    box off;
    %title('\boldmath$\hat{f}_j=f_j\circ H^{-1}$','interpreter','latex','FontSize',70)
    title('\boldmath$\tilde{f}_j=f_j\circ H^{-1}$','interpreter','latex')
    saveas(figure,'F:\Research\Project\1_Filtered_trajectory_recovery\Fig\inverse.png');
end
end

function [xs, s1] = deconvolution_without_known_minmax(F_x, F_y, mu, sigma, nu, ifplot)
lambda = 20;
% use less points for deconvolution 
x_min = min(F_x);
x_max = max(F_x);
n = 100;

% sigma can be at most (xmax - xmin)/6 since for a constant trajectory, the
% gaussian ranges from (-3*sigma, 3*sigma)
sigma = min(sigma, (x_max - x_min)/6);
if x_min < 0
    sigma = min(sigma, (0 - x_min)/3);
end

L1 = calc_Laplacian_1d(ones(1, n-1));
L1 = L1(2:end-1,:);

xs = linspace(0, x_max, n);

delta_x = xs(2) - xs(1);
[PSF_y, PSF_x] = create_gaussian_psf(delta_x, mu, sigma);
%[PSF_y, PSF_x] = create_t_psf(delta_x, mu, sigma, nu);

if ifplot == 1
    figure;
    plot(PSF_x,PSF_y,'black')
    set(gca,'fontweight','bold')
    %plot(PSF_x,PSF_y,'black','LineWidth',10)
    %set(gca,'fontweight','bold','FontSize',70,'LineWidth',10)
    %axis tight;
    %set(gca,'xticklabel',[],'yticklabel',[])
    box off;
    %title('\boldmath$\mathcal{N}(0,\sigma^{2})$','interpreter','latex','FontSize',70)
    title('\boldmath$\mathcal{N}(0,\sigma^{2})$','interpreter','latex')
    saveas(figure,'F:\Research\Project\1_Filtered_trajectory_recovery\Fig\filter.png');
end

l = floor(length(PSF_y) / 2);

% construct Fy for deconvolution
Fx_left = delta_x * (-l+1:0) + xs(1);
Fx_right = xs(2:end);
Fx = [Fx_left, Fx_right];
Fy = interp1(F_x, F_y, Fx, 'linear', NaN);

n = length(Fy);
% set the outside points to 0 or 1 based on whether the points are on the
% left side or the right side
%Fy(isnan(Fy) & (1:length(Fy)) < length(Fy)/2) = 0;
%Fy(isnan(Fy) & (1:length(Fy)) > length(Fy)/2) = 1;
Fy(isnan(Fy)) = 0;

%l0 = max(l+1, floor((0 - x_min)/delta_x*2));
%l0 = min(l0,3*l);
l0 = 2*l;

% [Fy2, ksi] = do_fft_1d(Fx, Fy);
% gf = -2*pi^2*ksi.^2*sigma^2;
% figure, plot(ksi, log(abs(Fy2)));
% hold on;
% plot(ksi, gf);

% create the kernel and b
K = zeros(n, n+2*l);
for i = 1:n
    K(i, i-1 + (1:2*l+1)) = PSF_y;
end

l1 = search_for_l1(K, l0, L1, Fx, Fy, n, l, lambda);

s1 = calc_s(K, L1, Fx, Fy, n, l, l0, l1, lambda);
xs = Fx(l0-l+1:end-l1+l);

%s1 = (s1-s1(1))/(s1(end)-s1(1));
%disp(['Start of stages: ', num2str(s1(1))]);
s1 = [0, s1', 1];
xs = [Fx(l0-l), xs, Fx(end-l1+l+1)];

end

function l1 = search_for_l1(K, l0, L1, Fx, Fy, n, l, lambda)
% search for l1

% l1 can be at most n/2 + l
t = min(3*l,n+2*l-l0-2);
lins = linspace(l+1,max(t,l+2),5);
l1s = round(lins);
%l1s = [l+1:max(t,l+2)];

obj_vals = zeros(1, length(l1s));
for idx = 1:length(l1s)
    l1 = l1s(idx);
    
    obj_val = eval_obj_fun(K, L1, Fx, Fy, n, l, l0, l1, lambda);    
    
    obj_vals(idx) = obj_val;
end

[~, ind] = min(obj_vals);
l1 = l1s(ind);
end

function obj_val = eval_obj_fun(K, L1, Fx, Fy, n, l, l0, l1, lambda)
[s, A, b, L, lambda] = calc_s(K, L1, Fx, Fy, n, l, l0, l1, lambda);

% objective
obj_val = sum((A*s - b).^2) + lambda * sum((L*s).^2);
end

function [s, A, b, L, lambda] = calc_s(K, L1, Fx, Fy, n, l, l0, l1, lambda)
K2 = K(:,l0+1:end-l1);
K3 = K(:,end-l1+1:end);

b = Fy' - K3*ones(l1,1);
xs = Fx;
A = K2;

% create L
L = L1(1:n+2*l-l0-l1-2,1:n+2*l-l0-l1);

A1A = A'*A;
% lambda = max(A1A(:)) * lambda;

%% fmincon/quadprog optimization
%fun = @(x) norm(A*x-b,2);
m = n+2*l-l0-l1;
D = create_difference(m);
x0 = linspace(0,1,m);
x0 = x0';
%c = zeros(m-1,1);
c = -0.001*ones(m-1,1);
options = optimoptions('quadprog','Display','off');
%s = quadprog(A'*A + lambda*(L'*L),-A'*b,D,c,[],[],[],[],x0,options);
s = quadprog(A'*A + lambda*(L'*L),-A'*b,D,c,[],[],0.001*ones(m,1),0.999*ones(m,1),[],options);
%s = quadprog(A'*A,-A'*b,D,c,[],[],0.001*ones(m,1),0.999*ones(m,1),[],options);
%display(A)
%s = fmincon(fun,x0,D,c);

%% GRBF
% create the RBF kernel
%rbf_K = create_RBF_kernel(xs);
%K = rbf_K' * A;
%s = quadprog(K'*K,-A'*b,D,c);
%% Laplacian Regularization
%s = inv(A'*A + lambda*L'*L)*A'*b;
end

function D = create_difference(n)
D = zeros(n-1,n);
for i = 1:n-1
    D(i,i) = 1;
    D(i,i+1) = -1;
end
end

function [xs, s1] = deconvolution_with_known_minmax(F_x, F_y, xmin, xmax, sigma)
% use less points for deconvolution 
rec_pts1 = 100;
xs = linspace(xmin, xmax, rec_pts1);
delta_x = xs(2) - xs(1);
[PSF_y, PSF_x] = create_gaussian_psf(delta_x, sigma);
l = floor(length(PSF_y) / 2);
n = rec_pts1 - 2;

% construct Fy for deconvolution
Fx_left = delta_x * (-l+1:0) + xs(1);
Fx_center = xs(2:end-1);
Fx_right = delta_x * (0:l-1) + xs(end);
Fx = [Fx_left, Fx_center, Fx_right];
Fy = interp1(F_x, F_y, Fx, 'linear', NaN);

% set the outside points to 0 or 1 based on whether the points are on the
% left side or the right side
Fy(isnan(Fy) & (1:length(Fy)) < length(Fy)/2) = 0;
Fy(isnan(Fy) & (1:length(Fy)) > length(Fy)/2) = 1;

% create the kernel and b
K = zeros(n+2*l, n);
for i = 1:n
    K(i-1 + (1:2*l+1), i) = fliplr(PSF_y);
end

b = zeros(n+2*l, 1);
tmp = cumsum(fliplr(PSF_y));
b(end-2*l+1:end) = tmp(1:end-1);

b = Fy' - b;

% create the RBF kernel
%rbf_K = create_RBF_kernel(xs(2:end-1));
%K = K * rbf_K;

L = calc_Laplacian_1d(ones(1, n-1));
L = L(2:end-1,:);

% solve for s
lambda = 10;
% tic
s1 = inv(K'*K + lambda*L'*L) * (K'*b);
% toc
% a1 = K \ b;

% s1 = rbf_K * a1;

% take the inverse
s1 = [0, s1', 1];
end

function rbf_K = create_RBF_kernel(xs1)
rbf_pts = 10;
rbf_sigma = (xs1(end) - xs1(1)) / (rbf_pts - 1);
centers = linspace(xs1(1), xs1(end), rbf_pts);
D = pdist2(centers',xs1','euclidean');
D = D';
rbf_K = exp(- D.^2 / (2*rbf_sigma^2));
end

function H = deconvolution(Fx, Fy, sigma)
if sigma == 0
    H = Fy;
    return;
end

if 1
    
    
elseif 0
    t = 0.5 * sigma^2;
    delta_t = 0.01;
    delta_x = Fx(2) - Fx(1);
    H = Fy;
    for iter = (0:delta_t:t)
        H = H - delta_t * ((H([1,1:end-1]) + H([2:end,end]) - 2*H)/(delta_x^2));
    end
elseif 0
	F_fr = fft_1d(Fy);
	varb = create_filter(Fx, sigma);
	H_fr = F_fr.*varb;
	H = inverse_fft_1d(H_fr);
	H = real(H);
else
    [PSF_y, PSF_x] = create_gaussian_psf(Fx(2) - Fx(1), sigma);

    % Fy = [zeros(size(Fy)), Fy, ones(size(Fy))];

    H = deconv(Fy, PSF_y);

    if 0 % wiener filter
        NSR = 0;
        H = deconvwnr(Fy, PSF_y, NSR);
    else
    %     H = deconvreg(Fy, PSF_y);
    end
end

end

function [Fy2, ksi] = do_fft_1d(Fx, Fy)
Fy1 = pad_cdf(Fy, length(Fy));
Fy2 = fft_1d(Fy1);

delta_x = Fx(2) - Fx(1);

N = length(Fy1);
freq_interval = 1 / (delta_x*N);

ksi = freq_interval*((-N/2):(N/2-1));
end

function fft_g = fft_1d(g)
% fft_g = fftshift(fft(ifftshift(g)));
fft_g = fftshift(fft(g));
end

function g1 = inverse_fft_1d(fft_g1)
% g1 = fftshift(ifft(ifftshift(fft_g1)));
g1 = ifft(ifftshift(fft_g1));
end


function Fy = pad_cdf(Fy, pad_size)

Fy = [zeros(1, pad_size), Fy, ones(1, pad_size)];
end

function Fy = unpad_cdf(Fy, pad_size)

Fy = Fy(pad_size+1:end-pad_size);
end


function varb = create_filter(Fx, sigma)
delta_xs = Fx(2:end) - Fx(1:end-1);
delta_x = mean(delta_xs);
N = length(Fx);
freq_interval = 1 / (delta_x*N);

ksi = freq_interval*((-N/2):(N/2-1));

varb = CDF_norm_fr_cons(ksi,sigma);

% suppress_radius = 10;
% varb(1:N/2+1-suppress_radius) = 1;
% varb(N/2+1+suppress_radius:end) = 1;

varb(varb > 3) = 1;
end


function y = CDF_norm_fr_cons(ksi,sigma)
y =  exp(2*pi^2*ksi.^2*sigma^2);
end
