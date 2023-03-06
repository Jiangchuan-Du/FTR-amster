cd 'C:\Users\86180\Desktop\FTR'
addpath("FTR_code\")

eval= readmatrix('.\Input\TADPOLE_test_ro.csv');
PTID = eval(:,1);
CN_list = eval(:,2)  == 1;
MCI_list = eval(:,2)  == 2;
AD_list = eval(:,2)  == 3;

mat = eval(:,3:end);
nsamp = size(mat,1);
nbiom = size(mat,2)-2;


%% Normalization 
CN_mat = mat(CN_list,:);
nor_mat = zeros(nsamp,nbiom);
for i = 1:nbiom
    pd = fitdist(CN_mat(:,i),'Normal');
    nor_mat(:,i) = (mat(:,i)-pd.mu)/pd.sigma; 
    if mean(nor_mat(:,i))<0
        nor_mat(:,i) = -1*nor_mat(:,i);
    end
end

% FTR
nsubtyp = 3;
max_iter = 50;
max_ep = 30;
num_int = 1000;

[nsamp,nbiom] = size(nor_mat);
traj = zeros(nbiom,num_int,nsubtyp,max_ep);
subtyp = zeros(nsamp,max_ep);
stag = zeros(nsamp,max_ep);
mean_sgm = zeros(max_ep,1);
xmin = quantile(nor_mat,0.001,1);
xmax = quantile(nor_mat,0.999,1);

for ep = 1:max_ep
    disp(ep)
    [traj(:,:,:,ep),subtyp(:,ep),stag(:,ep),mean_sgm(ep)]... 
         = kmeans_subtype(nor_mat,PTID,nsubtyp,max_iter,num_int,xmin,xmax,1);
end

[~,bes_ep] = min(mean_sgm);
bes_subtyp = subtyp(:,bes_ep);
best_traj = traj(:,:,:,bes_ep);
bes_stag = stag(:,bes_ep);

writematrix([PTID,bes_stag],'.\Output\stage.csv');
writematrix([PTID,bes_subtyp],'.\Output\subtype.csv');
for i = 1:nsubtyp
    writematrix(best_traj(:,:,i),['.\Output\trajectory',int2str(i),'.csv']);
end




