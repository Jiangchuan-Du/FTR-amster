function [traj,subtyp,stag,mean_sgm] = kmeans_subtype(dat,PTID,nsubtyp,max_iter,num_int,xmin,xmax,with_filter)

    [nsamp, nbiom] = size(dat);
    sigma = 0.2 * ones(nbiom,1);
    mu = zeros(nbiom,1);
    nu = ones(nbiom,1);
    subtyp = randi(nsubtyp,[nsamp 1]);
    traj = zeros(nbiom,num_int,nsubtyp);

    for iter = 1:max_iter
        for k = 1:nsubtyp
            dat_sub = dat(subtyp == k,:);
            for j = 1:nbiom
                if iter > 1
                    sigma(j) = sqrt(mean(dist_samp_min(:,j)));
                end
                ifplot = 0;
                if with_filter
                    [s,f] = ftr(dat_sub(:,j), xmin(j), xmax(j), num_int,  mu(j), sigma(j), nu(j), ifplot);
                else
                    [s,f] = ftr(dat_sub(:,j), xmin(j), xmax(j), num_int,  0, ifplot);
                end
                traj(j,:,k) = f;
            end
            re_traj = reparam_traj(traj(:,:,k)');
            traj(:,:,k) = re_traj';
        end

        dist_samp = zeros(nsamp,nbiom,nsubtyp);
        diff_samp = zeros(nsamp,nbiom,nsubtyp);
        stag_samp = zeros(nsamp,nsubtyp);
        for k = 1:nsubtyp
            [stag_samp(:,k),dist_samp(:,:,k),diff_samp(:,:,k)] = cal_dis(dat,traj(:,:,k));
        end

        if ~isempty(PTID)
            uni_id = unique(PTID);
            for i = 1:length(uni_id)
                rws = dist_samp(PTID==uni_id(i),:,:);
                dist_samp(PTID==uni_id(i),:,:) = repmat(mean(rws,1),sum(PTID==uni_id(i)),1,1);
            end
        end

        [~,subtyp] = min(sum(dist_samp,2),[],3);

        p = 1:nsamp;
        ind1 = sub2ind(size(stag_samp),p',subtyp);
        stag = stag_samp(ind1);
        
        mat1 = repmat(1:nsamp,nbiom,1).';
        mat2 = repmat(1:nbiom,nsamp,1);
        mat3 = repmat(subtyp,1,nbiom);
        ind2 = sub2ind(size(dist_samp),mat1(:),mat2(:),mat3(:));
        dist_samp_min = reshape(dist_samp(ind2),nsamp,nbiom);
        diff_samp_min = reshape(diff_samp(ind2),nsamp,nbiom);

        mu = zeros(nbiom,1);
        sigma = zeros(nbiom,1);
        nu = zeros(nbiom,1);
        for i = 1:nbiom
            %pd = fitdist(diff_samp_min(:,i),'tLocationScale');
            pd = fitdist(diff_samp_min(:,i),'normal');
            mu(i) = pd.mu;
            sigma(i) = pd.sigma;
            %nu(i) = pd.nu;
        end
 
        mean_sgm = mean(sigma);
        %mean_sgm = sigma;

        %if mod(iter, 50) == 0
            %elapsed_t = toc(start_t);
            %start_t = tic;
            %disp(['Elapsed time for iteration ',num2str(iter), ': ', num2str(elapsed_t)]);
            %disp(['Random index for iteration ',num2str(iter), ': ', num2str(subtype_rand_index(iter))]);
        %end
    end
    %pd = fitdist(diff_samp_min(:,1),'tLocationScale');
    %figure
    %qqplot(diff_samp_min(:,1))
    %figure
    %qqplot(diff_samp_min(:,1),pd)
    %disp(['Mean sigma for biomarkers:', num2str(mean_sgm)]);
end