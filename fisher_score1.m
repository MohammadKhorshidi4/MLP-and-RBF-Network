function score = fisher_score1(x)
N=length(x);
load('Normalized_Train_Features.mat');

Normalized_Train_Features1=Normalized_Train_Features(x,:);

load('Right_indices.mat');
load('Left_indices.mat');

N1 = length(Right_indices);
N2 = length(Left_indices);

Mu1 = mean(Normalized_Train_Features1(:,Right_indices),2);

Mu2 = mean(Normalized_Train_Features1(:,Left_indices),2);

Mu0 = mean(Normalized_Train_Features1,2);

S1 = zeros(N);
S2 = zeros(N);

for i = 1 : N1
    S1 = S1 + (Normalized_Train_Features1(:,Right_indices(i))- Mu1)*(Normalized_Train_Features1(:,Right_indices(i))- Mu1)';
end

S1 = S1/N1;
for i = 1 : N2
    S2 = S2 + (Normalized_Train_Features1(:,Left_indices(i))- Mu2)*(Normalized_Train_Features1(:,Left_indices(i))- Mu2)';
end

S2 = S2/N2;

Sw = S1 + S2;
Sb = (Mu1-Mu0)*(Mu1-Mu0)' + (Mu2-Mu0)*(Mu2-Mu0)';

J = trace(Sb)/(trace(Sw)+1e-20);

score =  -J;
