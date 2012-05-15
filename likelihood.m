% 対数尤度関数
function sum = likelihood(X, mean, cov, pi_k)
sum = 0.0;
for n=1:length(X)
    temp = 0.0;
    K = 2;  % 混合ガウス分布の数（固定）
    for k=1:K % K=2
        temp = temp + pi_k(1,k) * gaussian(X(n,:), mean(:,k), cov(:,:,k));
    end
    sum = sum + log(temp);
end