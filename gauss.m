clear all
close all
hold on

% �f�[�^�Z�b�g�̐ݒ�
K = 2  % �����K�E�X���z�̐��i�Œ�j
mu_1 = [0 0];
mu_2 = [10 10];
Sigma1 = [25 0; 0 25];
Sigma2 = [25 0; 0 25];
R_1 = chol(Sigma1);
R_2 = chol(Sigma2);
pi_1 = 0.4;
pi_2 = 0.6;
z_1 = repmat(mu_1,pi_1*1000,1) + randn(pi_1*1000,2)*R_1;
z_2 = repmat(mu_2,pi_2*1000,1) + randn(pi_2*1000,2)*R_2;
z = [z_1; z_2]; % z_1��z_2�𐂒��Ɍ���

%[z]=scale(z); % �f�[�^��W����
N = length(z) % �f�[�^��

% �P���f�[�^���獬���K�E�X���z�̃p�����[�^��EM�A���S���Y���Ő��肷��

% ���ρC���U�C�����W����������
mean = rand(2,K)
cov = zeros(2,2,K); % 2*2�s���K=2�̕��z�ɂ��čl����
for k=1:K
    cov(:,:,k) = [1.0 0.0; 0.0 1.0]; % �P�ʍs��
end
pi_k = rand(1,K) % 2����
% ���S���̋�z���p��
gamma = zeros(N,K); % N�̓f�[�^���i�����1000�j
% ������
% percentage = zeros(N);
right = 0.0;
wrong = 0.0;

% �ΐ��ޓx�̏����l���v�Z
like = likelihood(z, mean, cov, pi_k);

tic;
turn = 0;
while true

    turn % �o�͂��Ċm�F
    like % �o�͂��Ċm�F
    % E-step : ���݂̃p�����[�^���g���ĕ��S�����v�Z
    for n=1:N
        % �����k�ɂ��Ȃ��̂ōŏ���1�񂾂��v�Z
        denominator = 0.0;
        for j=1:K
            denominator = denominator + pi_k(1,j) * gaussian(z(n,:), mean(:,j), cov(:,:,j));
        end
        % �ek�ɂ��ĕ��S�����v�Z
        for k=1:K
            gamma(n,k) = pi_k(1,k) * gaussian(z(n,:), mean(:,k), cov(:,:,k)) / denominator;
        end
    end
    
    % M-step : ���݂̕��S�����g���ăp�����[�^���Čv�Z
    for k=1:K
        % Nk���v�Z����iNk��n(�f�[�^��)�ɂ��Ă̕��S���̑��a�j
        Nk = 0.0;
        for n=1:N
            Nk = Nk + gamma(n,k);
        end
        % ���ς��Čv�Z
        mean(:,k) = [0.0; 0.0];
        for n=1:N
            mean(:,k) = mean(:,k) + gamma(n,k)*z(n,:)'; % 2*1�̗�x�N�g���ɂȂ�悤��z��]�u
        end
        mean(:,k) = mean(:,k)/Nk;
        
        % �����U���Čv�Z
        cov(:,:,k) = [0.0 0.0; 0.0 0.0];
        for n=1:N
            temp = z(n,:)' - mean(:,k); %2*1�̗�x�N�g��
            cov(:,:,k) = cov(:,:,k) + gamma(n,k) * temp * temp'; % �c�x�N�g���~���x�N�g��
        end
        cov(:,:,k) = cov(:,:,k) / Nk;
        
        % �����W�����Čv�Z
        pi_k(1,k) = Nk / N;
        
    end
    
    % ��������
    new_like = likelihood(z, mean, cov, pi_k);
    diff = new_like - like;
    if (diff < 0.000000001)
        break;
    end
    like = new_like;
    turn = turn + 1;
    
end    
toc; % �v�Z���Ԃ̏o��

% ���ρC�����U�C���S�����o��
mean
cov
pi_k

% ��������`��
xlist = linspace(-20, 30, 50);
ylist = linspace(-20, 30, 50);
[X,Y] = meshgrid(xlist, ylist);
for k=1:K
    for row=1:50
        for column=1:50
            %Z = bivariate_normal(X, Y, sqrt(cov[k,0,0]), sqrt(cov[k,1,1]), mean[k,0], mean[k,1], cov[k,0,1]);
            %Z = gaussian(XY, mean(:,k), cov(:,:,k));
            tmp1 = 1 / (2*pi);
            tmp2 = 1 / (det(cov(:,:,k))^0.5);
            deviation = [X(row,column)-mean(1,k); Y(row,column)-mean(2,k)]; % �c�x�N�g��
            tmp3 = -0.5*deviation'*inv(cov(:,:,k))*deviation;
            Z(row,column) = tmp1 * tmp2 * exp(tmp3);
        end
    end
    contour(X, Y, Z);
end
xlim([-20,30]);
ylim([-20,30]);

% �v���b�g
%plot(z_1(:,1),z_1(:,2),'b+',z_2(:,1),z_2(:,2),'r+', 0, 0 ,'x')
plot(z_1(:,1),z_1(:,2),'b+',z_2(:,1),z_2(:,2),'r+', mean(1,1), mean(2,1), 'go', mean(1,2), mean(2,2), 'gx')
xlabel('x')
ylabel('y')
hold off;

% �������i1000�̃f�[�^�̂��������𐳂������z�Ɏ��ʂł������j
for n=1:N % 1~1000
    % n=1~400�͕��z�P�ɑ����Ă���
    if (n>=1)&(n<=0.4*N)
        if gamma(n,1)>=gamma(n,2) % ���S�����傫����ΐ��������ʂł��Ă���Ƃ���
            right = right + 1;
        else
            wrong = wrong + 1;
        end
    end
    % n=401~1000�͕��z�Q�ɑ����Ă���
    if (n>=0.4*N+1)&(n<=N)
        if gamma(n,1)<=gamma(n,2)
            right = right + 1;
        else
            wrong = wrong + 1;
        end        
    end
end
right
wrong
