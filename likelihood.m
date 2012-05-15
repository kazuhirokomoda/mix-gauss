% �ΐ��ޓx�֐�
function sum = likelihood(X, mean, cov, pi_k)
sum = 0.0;
for n=1:length(X)
    temp = 0.0;
    K = 2;  % �����K�E�X���z�̐��i�Œ�j
    for k=1:K % K=2
        temp = temp + pi_k(1,k) * gaussian(X(n,:), mean(:,k), cov(:,:,k));
    end
    sum = sum + log(temp);
end