function [PSF_y, PSF_x] = create_gaussian_psf(delta_x, mu, sigma)
%CREATE_GAUSSIAN_PSF Summary of this function goes here
%   Detailed explanation goes here

size = ceil(3*sigma / delta_x);
PSF_x = (-size:size)*delta_x;
%PSF_y = exp(-(PSF_x-mu).^2 / (2*sigma^2));
PSF_y = exp(-PSF_x.^2 / (2*sigma^2));
PSF_y = PSF_y / sum(PSF_y);

end

