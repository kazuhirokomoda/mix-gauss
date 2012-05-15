function result = gaussian(X, mean, cov)
% ‘½•Ï—ÊƒKƒEƒXŠÖ”
temp1 = 1 / ((2*pi)^(length(X)/2.0));
temp2 = 1 / (det(cov)^0.5);
temp3 = -0.5*(X'-mean)'*inv(cov)*(X'-mean);
result = temp1 * temp2 * exp(temp3);
end