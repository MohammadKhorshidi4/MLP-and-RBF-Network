%%
%MLP with ordinary feature selection with fisher score
clc
clear
L=50;
Fs=1000;

load('All_data.mat');
for i=1:316
    tes(:,:)=x_train(:,:,i);
    va=var(tes);
    ku=kurtosis(tes);
    sk=skewness(tes);
    di1=diff(tes);
    di2=diff(di1);
    va1=var(di1);
    va2=var(di2);
    ff=(sqrt(va2)./sqrt(va1))./(sqrt(va1)./sqrt(va));
    tr=1;
    rty=1;
    cv=[];
    for j=28:-1:1
        cv1=tes(:,j);
        for q=1:j
            cv2=tes(:,q);
            cv3=0.02*sum(((cv2-mean(cv2))).*((cv1-mean(cv1))));
            cv(1,tr)=cv3;
            cv4=cov(cv1,cv2);
            cv5(1,rty)=cv4(1,2);
            rty=rty+1;
            cv5(1,rty)=cv4(2,1);
            rty=rty+1;
            tr=tr+1;
        end
    end
    mef=meanfreq(tes,1000);
    mdf=medfreq(tes,1000);
    bw=obw(tes,1000);
    delpw=bandpower(tes,1000,[1 3.5]);
    thpw=bandpower(tes,1000,[4 7.5]);
    alpw=bandpower(tes,1000,[8 13]);
    betpw=bandpower(tes,1000,[14 30]);
    for rt=1:28
        entre(rt)=wentropy(tes(:,1),'shannon');
    end
    for ui=1:28
        Y=fft(tes(:,ui));
        f = Fs*(0:(L/2))/L;
        P2 = abs(Y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        [mf,f11]=max(P1);
        fr(ui)=f(f11);
    end
    af=[va,ff,mdf,bw,delpw,thpw,alpw,betpw,cv,fr];
    feat(:,i)=af;
end
[nfeat,xPS] = mapminmax(feat) ;

%%
l1=find(y_train==1);
l0=find(y_train==0);
for i=1:length(feat(:,1))
    fea1=nfeat(i,l1);
    fea0=nfeat(i,l0);
    v1=var(fea1);
    v0=var(fea0);
    m1=mean(fea1);
    m0=mean(fea0);
    m2=mean(nfeat(i,:));
    jf(i,1)=((abs(m2-m1))^2+(abs(m2-m0))^2)/(v1+v0);
end
jf=abs(jf);
[M1,I1]=maxk(jf,30);

%%
idfeat=nfeat(I1,:);
%%
tr=1;
ACCma=0;
for N1=1:10
    for N2=1:10
        for N3=1:10
            ACC = 0 ;
            for k=1:4
                train_indices = [1:(k-1)*79,k*79+1:316] ;
                valid_indices = (k-1)*79+1:k*79 ;

                TrainX = idfeat(:,train_indices) ;
                ValX = idfeat(:,valid_indices) ;
                TrainY = y_train(train_indices) ;
                ValY = y_train(valid_indices) ;

            % feedforwardnet, newff, paternnet
            % patternnet(hiddenSizes,trainFcn,performFcn)
                net = feedforwardnet([N1 N2 N3],'trainbr');
                net1 = train(net,TrainX,TrainY);

                predict_y = net1(ValX);
%         Thr = 0.5 ;
        
                p_TrainY = net1(TrainX);
                [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
                Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;
        
                predict_y = predict_y >= Thr ;

                ACC = ACC + length(find(predict_y==ValY)) ;
            end
            
            ACCMat(tr)=ACC/316;
            ACCma1=ACCMat(tr);
            if ACCma1>ACCma
                ACCma=ACCma1;
                save('idealNet','net1');
            end
            tr=tr+1;
        end
    end
end

%%
load('idealNet.mat');
ACCma=0;
tr=1;
while ACCma<0.62  
    ACC = 0 ;
    
        for k=1:4
            train_indices = [1:(k-1)*79,k*79+1:316] ;
            valid_indices = (k-1)*79+1:k*79 ;
            TrainX = idfeat(:,train_indices) ;
            ValX = idfeat(:,valid_indices) ;
            TrainY = y_train(train_indices);
            ValY = y_train(valid_indices) ;

            % feedforwardnet, newff, paternnet
            % patternnet(hiddenSizes,trainFcn,performFcn)
            
            

            predict_y = net1(ValX);
%         Thr = 0.5 ;
        
            p_TrainY = net1(TrainX);
            [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
            Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;
        
            predict_y = predict_y >= Thr ;

            ACC = ACC + length(find(predict_y==ValY)) ;
        end
        ACCma1=ACC/316;
        if ACCma1>ACCma
            ACCma=ACCma1;
        end
        tr=tr+1;
end


%%
for i=1:100
    tes(:,:)=x_test(:,:,i);
    va=var(tes);
    ku=kurtosis(tes);
    sk=skewness(tes);
    di1=diff(tes);
    di2=diff(di1);
    va1=var(di1);
    va2=var(di2);
    ff=(sqrt(va2)./sqrt(va1))./(sqrt(va1)./sqrt(va));
    tr=1;
    rty=1;
    cv=[];
    for j=28:-1:1
        cv1=tes(:,j);
        for q=1:j
            cv2=tes(:,q);
            cv3=0.02*sum(((cv2-mean(cv2))).*((cv1-mean(cv1))));
            cv(1,tr)=cv3;
            cv4=cov(cv1,cv2);
            cv5(1,rty)=cv4(1,2);
            rty=rty+1;
            cv5(1,rty)=cv4(2,1);
            rty=rty+1;
            tr=tr+1;
        end
    end
    mef=meanfreq(tes,1000);
    mdf=medfreq(tes,1000);
    bw=obw(tes,1000);
    delpw=bandpower(tes,1000,[1 3.5]);
    thpw=bandpower(tes,1000,[4 7.5]);
    alpw=bandpower(tes,1000,[8 13]);
    betpw=bandpower(tes,1000,[14 30]);
    for rt=1:28
        entre(rt)=wentropy(tes(:,1),'shannon');
    end
    for ui=1:28
        Y=fft(tes(:,ui));
        f = Fs*(0:(L/2))/L;
        P2 = abs(Y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        [mf,f11]=max(P1);
        fr(ui)=f(f11);
    end
    af=[va,ff,mdf,bw,delpw,thpw,alpw,betpw,cv,fr];
    feattest(:,i)=af;
end
[nfeattest,xPS] = mapminmax(feattest) ;

idfeattest=nfeattest(I1,:);

%%
load('idealNet.mat');
predict_test=net1(idfeattest);
predict_test=predict_test>= Thr;
save('MLPOUTPUT','predict_test');

%%
%RBF with ordinary feature selection with fisher score

clc
clear
L=50;
Fs=1000;

load('All_data.mat');
for i=1:316
    tes(:,:)=x_train(:,:,i);
    va=var(tes);
    ku=kurtosis(tes);
    sk=skewness(tes);
    di1=diff(tes);
    di2=diff(di1);
    va1=var(di1);
    va2=var(di2);
    ff=(sqrt(va2)./sqrt(va1))./(sqrt(va1)./sqrt(va));
    tr=1;
    rty=1;
    cv=[];
    for j=28:-1:1
        cv1=tes(:,j);
        for q=1:j
            cv2=tes(:,q);
            cv3=0.02*sum(((cv2-mean(cv2))).*((cv1-mean(cv1))));
            cv(1,tr)=cv3;
            cv4=cov(cv1,cv2);
            cv5(1,rty)=cv4(1,2);
            rty=rty+1;
            cv5(1,rty)=cv4(2,1);
            rty=rty+1;
            tr=tr+1;
        end
    end
    mef=meanfreq(tes,1000);
    mdf=medfreq(tes,1000);
    bw=obw(tes,1000);
    delpw=bandpower(tes,1000,[1 3.5]);
    thpw=bandpower(tes,1000,[4 7.5]);
    alpw=bandpower(tes,1000,[8 13]);
    betpw=bandpower(tes,1000,[14 30]);
    for rt=1:28
        entre(rt)=wentropy(tes(:,1),'shannon');
    end
    for ui=1:28
        Y=fft(tes(:,ui));
        f = Fs*(0:(L/2))/L;
        P2 = abs(Y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        [mf,f11]=max(P1);
        fr(ui)=f(f11);
    end
    af=[va,ff,mdf,bw,delpw,thpw,alpw,betpw,cv,fr];
    feat(:,i)=af;
end
[nfeat,xPS] = mapminmax(feat) ;

%%
l1=find(y_train==1);
l0=find(y_train==0);
for i=1:length(feat(:,1))
    fea1=nfeat(i,l1);
    fea0=nfeat(i,l0);
    v1=var(fea1);
    v0=var(fea0);
    m1=mean(fea1);
    m0=mean(fea0);
    m2=mean(nfeat(i,:));
    jf(i,1)=((abs(m2-m1))^2+(abs(m2-m0))^2)/(v1+v0);
end
jf=abs(jf);
[M1,I1]=maxk(jf,30);

%%
idfeat=nfeat(I1,:);
%%
spreadMat = 0.1:0.1:2 ;
NMat = 1:100 ;
tr=1;
for N1=1:length(spreadMat)
    spread = spreadMat(N1) ;
    for N2=1:100
        Maxnumber = NMat(N2) ;
         ACC = 0 ;
         for k=1:4
            train_indices = [1:(k-1)*79,k*79+1:316] ;
            valid_indices = (k-1)*79+1:k*79 ;

            TrainX = idfeat(:,train_indices) ;
            ValX = idfeat(:,valid_indices) ;
            TrainY = y_train(train_indices);
            ValY = y_train(valid_indices) ;

            % feedforwardnet, newff, paternnet
            % patternnet(hiddenSizes,trainFcn,performFcn)
            
            net = newrb(TrainX,TrainY,0,spread,Maxnumber) ;
            predict_y = net(ValX);
%         Thr = 0.5 ;
        
            p_TrainY = net(TrainX);
            [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
            Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;
        
            predict_y = predict_y >= Thr ;
            ACC = ACC + length(find(predict_y==ValY)) ;
        end
        ACCMat(tr)=ACC/316;
        tr=tr+1;
    end
end


%%
[io,op]=max(ACCMat);
spr=ceil(op/100);
nm=op-(spr-1)*100;
%%
for i=1:100
    tes(:,:)=x_test(:,:,i);
    va=var(tes);
    ku=kurtosis(tes);
    sk=skewness(tes);
    di1=diff(tes);
    di2=diff(di1);
    va1=var(di1);
    va2=var(di2);
    ff=(sqrt(va2)./sqrt(va1))./(sqrt(va1)./sqrt(va));
    tr=1;
    rty=1;
    cv=[];
    for j=28:-1:1
        cv1=tes(:,j);
        for q=1:j
            cv2=tes(:,q);
            cv3=0.02*sum(((cv2-mean(cv2))).*((cv1-mean(cv1))));
            cv(1,tr)=cv3;
            cv4=cov(cv1,cv2);
            cv5(1,rty)=cv4(1,2);
            rty=rty+1;
            cv5(1,rty)=cv4(2,1);
            rty=rty+1;
            tr=tr+1;
        end
    end
    mef=meanfreq(tes,1000);
    mdf=medfreq(tes,1000);
    bw=obw(tes,1000);
    delpw=bandpower(tes,1000,[1 3.5]);
    thpw=bandpower(tes,1000,[4 7.5]);
    alpw=bandpower(tes,1000,[8 13]);
    betpw=bandpower(tes,1000,[14 30]);
    for rt=1:28
        entre(rt)=wentropy(tes(:,1),'shannon');
    end
    for ui=1:28
        Y=fft(tes(:,ui));
        f = Fs*(0:(L/2))/L;
        P2 = abs(Y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        [mf,f11]=max(P1);
        fr(ui)=f(f11);
    end
    af=[va,ff,mdf,bw,delpw,thpw,alpw,betpw,cv,fr];
    feattest(:,i)=af;
end
[nfeattest,xPS] = mapminmax(feattest) ;

idfeattest=nfeattest(I1,:);

%%
ACCma1=0;
spread=spr;
Maxnumber=nm+58;
while ACCma1<0.62
    ACC=0;
    for k=1:4
        train_indices = [1:(k-1)*79,k*79+1:316] ;
        valid_indices = (k-1)*79+1:k*79 ;
        TrainX = idfeat(:,train_indices) ;
        ValX = idfeat(:,valid_indices) ;
        TrainY = y_train(train_indices);
        ValY = y_train(valid_indices) ;

     % feedforwardnet, newff, paternnet
     % patternnet(hiddenSizes,trainFcn,performFcn         
        net = newrb(TrainX,TrainY,0,spread,Maxnumber) ;
        predict_y = net(ValX);
%         Thr = 0.5 ;
        
        p_TrainY = net(TrainX);
        [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
        Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;
        
        predict_y = predict_y >= Thr ;
        ACC = ACC + length(find(predict_y==ValY)) ;
    end
    ACCma1=ACC/316;
end
predict_test=net(idfeattest);
predict_test = predict_test >= Thr ;
save('RBFOUTPUT','predict_test');
%%
save("RBFNETWORK","net");


%%
%%MLP and RBF with feature selection with genetic algorithm with fisher score

clc
clear
L=50;
Fs=1000;

load('All_data.mat');
for i=1:316
    tes(:,:)=x_train(:,:,i);
    va=var(tes);
    ku=kurtosis(tes);
    sk=skewness(tes);
    di1=diff(tes);
    di2=diff(di1);
    va1=var(di1);
    va2=var(di2);
    ff=(sqrt(va2)./sqrt(va1))./(sqrt(va1)./sqrt(va));
    tr=1;
    rty=1;
    cv=[];
    for j=28:-1:1
        cv1=tes(:,j);
        for q=1:j
            cv2=tes(:,q);
            cv3=0.02*sum(((cv2-mean(cv2))).*((cv1-mean(cv1))));
            cv(1,tr)=cv3;
            cv4=cov(cv1,cv2);
            cv5(1,rty)=cv4(1,2);
            rty=rty+1;
            cv5(1,rty)=cv4(2,1);
            rty=rty+1;
            tr=tr+1;
        end
    end
    mef=meanfreq(tes,1000);
    mdf=medfreq(tes,1000);
    bw=obw(tes,1000);
    delpw=bandpower(tes,1000,[1 3.5]);
    thpw=bandpower(tes,1000,[4 7.5]);
    alpw=bandpower(tes,1000,[8 13]);
    betpw=bandpower(tes,1000,[14 30]);
    for rt=1:28
        entre(rt)=wentropy(tes(:,1),'shannon');
    end
    for ui=1:28
        Y=fft(tes(:,ui));
        f = Fs*(0:(L/2))/L;
        P2 = abs(Y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        [mf,f11]=max(P1);
        fr(ui)=f(f11);
    end
    af=[va,sk,ku,ff,mef,mdf,bw,delpw,thpw,alpw,betpw,cv,cv5,fr,entre];
    feat(:,i)=af;
end
[nfeat,xPS] = mapminmax(feat) ;

%%
Right_indices=find(y_train==1);
Left_indices=find(y_train==0);
for i=1:length(feat(:,1))
    fea1=nfeat(i,Right_indices);
    fea0=nfeat(i,Left_indices);
    v1=var(fea1);
    v0=var(fea0);
    m1=mean(fea1);
    m0=mean(fea0);
    m2=mean(nfeat(i,:));
    jf(i,1)=((abs(m2-m1))^2+(abs(m2-m0))^2)/(v1+v0);
end
jf=abs(jf);
[M1,I1]=maxk(jf,100);

%%
idfeat1=nfeat(I1,:);
Normalized_Train_Features=idfeat1;
Number_of_grouped_features=30;
save('Normalized_Train_Features','Normalized_Train_Features');
save('Right_indices','Right_indices');
save('Left_indices','Left_indices');
%%
[x,fval,exitflag,output,population,score] = OPTIMIZEE(30,1,100);
%%
idfeat=idfeat1(x,:);
%%
%MLP
tr=1;
ACCma=0;
for N1=1:10
    for N2=1:10
        for N3=1:10
            ACC = 0 ;
            for k=1:4
                train_indices = [1:(k-1)*79,k*79+1:316] ;
                valid_indices = (k-1)*79+1:k*79 ;

                TrainX = idfeat(:,train_indices) ;
                ValX = idfeat(:,valid_indices) ;
                TrainY = y_train(train_indices) ;
                ValY = y_train(valid_indices) ;

            % feedforwardnet, newff, paternnet
            % patternnet(hiddenSizes,trainFcn,performFcn)
                net = feedforwardnet([N1 N2 N3],'trainbr');
                net1 = train(net,TrainX,TrainY);

                predict_y = net1(ValX);
%         Thr = 0.5 ;
        
                p_TrainY = net1(TrainX);
                [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
                Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;
        
                predict_y = predict_y >= Thr ;

                ACC = ACC + length(find(predict_y==ValY)) ;
            end
            
            ACCMat(tr)=ACC/316;
            ACCma1=ACCMat(tr);
            if ACCma1>ACCma
                ACCma=ACCma1;
                save('idealNetGENETIC','net1');
                save('THRGEN','Thr');
            end
            tr=tr+1;
        end
    end
end
ACCMAX=max(ACCMat);


%%
for i=1:100
    tes(:,:)=x_test(:,:,i);
    va=var(tes);
    ku=kurtosis(tes);
    sk=skewness(tes);
    di1=diff(tes);
    di2=diff(di1);
    va1=var(di1);
    va2=var(di2);
    ff=(sqrt(va2)./sqrt(va1))./(sqrt(va1)./sqrt(va));
    tr=1;
    rty=1;
    cv=[];
    for j=28:-1:1
        cv1=tes(:,j);
        for q=1:j
            cv2=tes(:,q);
            cv3=0.02*sum(((cv2-mean(cv2))).*((cv1-mean(cv1))));
            cv(1,tr)=cv3;
            cv4=cov(cv1,cv2);
            cv5(1,rty)=cv4(1,2);
            rty=rty+1;
            cv5(1,rty)=cv4(2,1);
            rty=rty+1;
            tr=tr+1;
        end
    end
    mef=meanfreq(tes,1000);
    mdf=medfreq(tes,1000);
    bw=obw(tes,1000);
    delpw=bandpower(tes,1000,[1 3.5]);
    thpw=bandpower(tes,1000,[4 7.5]);
    alpw=bandpower(tes,1000,[8 13]);
    betpw=bandpower(tes,1000,[14 30]);
    for rt=1:28
        entre(rt)=wentropy(tes(:,1),'shannon');
    end
    for ui=1:28
        Y=fft(tes(:,ui));
        f = Fs*(0:(L/2))/L;
        P2 = abs(Y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        [mf,f11]=max(P1);
        fr(ui)=f(f11);
    end
    af=[va,ff,mdf,bw,delpw,thpw,alpw,betpw,cv,fr];
    feattest(:,i)=af;
end
[nfeattest,xPS] = mapminmax(feattest) ;

idfeattest=nfeattest(x,:);

%%
load('idealNetGENETIC.mat');
load('THRGEN.mat');
predict_test=net1(idfeattest);
predict_test=predict_test>= Thr;
save('MLPOUTPUTGENETIC','predict_test');
%%

%RBF Network Training for The New Featurs.
spreadMat = 0.1:0.1:2 ;
NMat = 1:100 ;
tr=1;
ACCma=0;
for N1=1:length(spreadMat)
    spread = spreadMat(N1) ;
    for N2=1:100
        Maxnumber = NMat(N2) ;
         ACC = 0 ;
         for k=1:4
            train_indices = [1:(k-1)*79,k*79+1:316] ;
            valid_indices = (k-1)*79+1:k*79 ;

            TrainX = idfeat(:,train_indices) ;
            ValX = idfeat(:,valid_indices) ;
            TrainY = y_train(train_indices);
            ValY = y_train(valid_indices) ;

            % feedforwardnet, newff, paternnet
            % patternnet(hiddenSizes,trainFcn,performFcn)
            
            net = newrb(TrainX,TrainY,0,spread,Maxnumber) ;
            predict_y = net(ValX);
%         Thr = 0.5 ;
        
            p_TrainY = net(TrainX);
            [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
            Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;
        
            predict_y = predict_y >= Thr ;
            ACC = ACC + length(find(predict_y==ValY)) ;
        end
        ACCMat(tr)=ACC/316;
        ACCma1=ACCMat(tr);
        if ACCma1>ACCma
            ACCma=ACCma1;
            save('RBFNETWORKGENETIC','net');
        end
        tr=tr+1;
    end
end

%%
%Finding Best Threshold.
load('RBFNETWORKGENETIC.mat');
while ACCma1<0.62
    ACC=0;
    for k=1:4
        train_indices = [1:(k-1)*79,k*79+1:316] ;
        valid_indices = (k-1)*79+1:k*79 ;
        TrainX = idfeat(:,train_indices) ;
        ValX = idfeat(:,valid_indices) ;
        TrainY = y_train(train_indices);
        ValY = y_train(valid_indices) ;

     % feedforwardnet, newff, paternnet
     % patternnet(hiddenSizes,trainFcn,performFcn         
        predict_y = net(ValX);
%         Thr = 0.5 ;
        
        p_TrainY = net(TrainX);
        [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
        Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;
        
        predict_y = predict_y >= Thr ;
        ACC = ACC + length(find(predict_y==ValY)) ;
    end
    ACCma1=ACC/316;
end
%%
%Test Data Feature ExtracTion And Selection
for i=1:100
    tes(:,:)=x_test(:,:,i);
    va=var(tes);
    ku=kurtosis(tes);
    sk=skewness(tes);
    di1=diff(tes);
    di2=diff(di1);
    va1=var(di1);
    va2=var(di2);
    ff=(sqrt(va2)./sqrt(va1))./(sqrt(va1)./sqrt(va));
    tr=1;
    rty=1;
    cv=[];
    for j=28:-1:1
        cv1=tes(:,j);
        for q=1:j
            cv2=tes(:,q);
            cv3=0.02*sum(((cv2-mean(cv2))).*((cv1-mean(cv1))));
            cv(1,tr)=cv3;
            cv4=cov(cv1,cv2);
            cv5(1,rty)=cv4(1,2);
            rty=rty+1;
            cv5(1,rty)=cv4(2,1);
            rty=rty+1;
            tr=tr+1;
        end
    end
    mef=meanfreq(tes,1000);
    mdf=medfreq(tes,1000);
    bw=obw(tes,1000);
    delpw=bandpower(tes,1000,[1 3.5]);
    thpw=bandpower(tes,1000,[4 7.5]);
    alpw=bandpower(tes,1000,[8 13]);
    betpw=bandpower(tes,1000,[14 30]);
    for rt=1:28
        entre(rt)=wentropy(tes(:,1),'shannon');
    end
    for ui=1:28
        Y=fft(tes(:,ui));
        f = Fs*(0:(L/2))/L;
        P2 = abs(Y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        [mf,f11]=max(P1);
        fr(ui)=f(f11);
    end
    af=[va,ff,mdf,bw,delpw,thpw,alpw,betpw,cv,fr];
    feattest(:,i)=af;
end
[nfeattest,xPS] = mapminmax(feattest) ;

idfeattest=nfeattest(x,:);

%%
predict_test=net(idfeattest);
predict_test = predict_test >= Thr ;
save('RBFOUTPUTGENETIC','predict_test');
