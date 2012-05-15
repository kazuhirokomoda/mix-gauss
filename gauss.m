clear all
close all
hold on

% データセットの設定
K = 2  % 混合ガウス分布の数（固定）
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
z = [z_1; z_2]; % z_1とz_2を垂直に結合

%[z]=scale(z); % データを標準化
N = length(z) % データ数

% 訓練データから混合ガウス分布のパラメータをEMアルゴリズムで推定する

% 平均，分散，混合係数を初期化
mean = rand(2,K)
cov = zeros(2,2,K); % 2*2行列をK=2個の分布について考える
for k=1:K
    cov(:,:,k) = [1.0 0.0; 0.0 1.0]; % 単位行列
end
pi_k = rand(1,K) % 2次元
% 負担率の空配列を用意
gamma = zeros(N,K); % Nはデータ数（今回は1000）
% 正答率
% percentage = zeros(N);
right = 0.0;
wrong = 0.0;

% 対数尤度の初期値を計算
like = likelihood(z, mean, cov, pi_k);

tic;
turn = 0;
while true

    turn % 出力して確認
    like % 出力して確認
    % E-step : 現在のパラメータを使って負担率を計算
    for n=1:N
        % 分母はkによらないので最初に1回だけ計算
        denominator = 0.0;
        for j=1:K
            denominator = denominator + pi_k(1,j) * gaussian(z(n,:), mean(:,j), cov(:,:,j));
        end
        % 各kについて負担率を計算
        for k=1:K
            gamma(n,k) = pi_k(1,k) * gaussian(z(n,:), mean(:,k), cov(:,:,k)) / denominator;
        end
    end
    
    % M-step : 現在の負担率を使ってパラメータを再計算
    for k=1:K
        % Nkを計算する（Nkはn(データ数)についての負担率の総和）
        Nk = 0.0;
        for n=1:N
            Nk = Nk + gamma(n,k);
        end
        % 平均を再計算
        mean(:,k) = [0.0; 0.0];
        for n=1:N
            mean(:,k) = mean(:,k) + gamma(n,k)*z(n,:)'; % 2*1の列ベクトルになるようにzを転置
        end
        mean(:,k) = mean(:,k)/Nk;
        
        % 共分散を再計算
        cov(:,:,k) = [0.0 0.0; 0.0 0.0];
        for n=1:N
            temp = z(n,:)' - mean(:,k); %2*1の列ベクトル
            cov(:,:,k) = cov(:,:,k) + gamma(n,k) * temp * temp'; % 縦ベクトル×横ベクトル
        end
        cov(:,:,k) = cov(:,:,k) / Nk;
        
        % 混合係数を再計算
        pi_k(1,k) = Nk / N;
        
    end
    
    % 収束判定
    new_like = likelihood(z, mean, cov, pi_k);
    diff = new_like - like;
    if (diff < 0.000000001)
        break;
    end
    like = new_like;
    turn = turn + 1;
    
end    
toc; % 計算時間の出力

% 平均，共分散，負担率を出力
mean
cov
pi_k

% 等高線を描画
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
            deviation = [X(row,column)-mean(1,k); Y(row,column)-mean(2,k)]; % 縦ベクトル
            tmp3 = -0.5*deviation'*inv(cov(:,:,k))*deviation;
            Z(row,column) = tmp1 * tmp2 * exp(tmp3);
        end
    end
    contour(X, Y, Z);
end
xlim([-20,30]);
ylim([-20,30]);

% プロット
%plot(z_1(:,1),z_1(:,2),'b+',z_2(:,1),z_2(:,2),'r+', 0, 0 ,'x')
plot(z_1(:,1),z_1(:,2),'b+',z_2(:,1),z_2(:,2),'r+', mean(1,1), mean(2,1), 'go', mean(1,2), mean(2,2), 'gx')
xlabel('x')
ylabel('y')
hold off;

% 正答率（1000個のデータのうちいくつを正しい分布に識別できたか）
for n=1:N % 1~1000
    % n=1~400は分布１に属している
    if (n>=1)&(n<=0.4*N)
        if gamma(n,1)>=gamma(n,2) % 負担率が大きければ正しく識別できているとする
            right = right + 1;
        else
            wrong = wrong + 1;
        end
    end
    % n=401~1000は分布２に属している
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
