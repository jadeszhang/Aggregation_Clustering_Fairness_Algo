%% bigraph function

%%% Introduced trimmed mean
%%% Input
% DataInAll: [counts, counts conditional one feature 1, counts conditional on feature 2, ...]
% lambda: [correlation weights, distance weights, bias weights]
% Sthre:  bias mapping threshold
% caseI: index for using which measurement

%% caseI(1): handling of correlation
% caseI(1) = 1: set as 0
% caseI(1) = 2: set as the average of the non zeros
% caseI(1) = 3: set as 1

%% caseI(2): handling of conditional probability (UPDATED CONDITIONAL PROBABILITY)
% caseI{2}(1) = 1: set as 0 when denominator = 0
% caseI{2}(1) = 2: set as the mean(conditional prob ~=0)
% caseI{2}(1) = 3: set as 1 when denominator = 0

% caseI{2}(2) = 1: without pseudoBoolean
% caseI{2}(2) = 2: with pseudoBoolean
% caseI{2}(2) = 3: with pseudoBoolean but coefficient = conditional
% probability

%% caseI(3): handling of mapping bias from n-space to 1-space numerical
% caseI(3) = 1: average->mapping, all time points equally
% caseI(3) = 2: average->mapping, all time points but trimmed meean
% caseI(3) = 3: average->mapping, only the most recent time poitns
% caseI(3) = 4: mapping->average, all times points equally
% caseI(3) = 5: mapping->average, all time points but trimmed mean
% caseI(3) = 6: mapping->average, only the most recent time poitns (same as
% in case 3)

%% caseI(4): coefficients
% caseI(4) = [c0,c1,c2];



%%% Output
% DataOut: counts after aggregation
% IndexOut: index for aggregation
% BiasM: bias metrics
% BiasV: bias value


function [DataOut, IndexOut, BiasM, BiasV] = bigraph(DataInAll,lambda,Sthre,caseI)            

    dDepth = size(DataInAll{1,1},2);
    rLength = size(DataInAll,2);                                           %% # of features
    yLength = size(DataInAll{1,1},1);                                      %% # of years
    DataIn0 = cell2mat(DataInAll);                                         %% all counts

    DataInAll = DataIn0(:,1:dDepth:end);
    CorM = corrcoef(DataInAll);

    switch caseI{1}                                                        %% BECAREFUL WITH THIS APPROXIMATION

        case 1
            CorM(isnan(CorM))  = deal(0);                                          
        case 2
            a0 = mean(CorM(~isnan(CorM)));
            CorM(isnan(CorM))  = deal(a0);
        case 3
            CorM(isnan(CorM))  = deal(1);  
    end
    clear a0



%% Bias measurement

    BiasM = cell(1,dDepth-1);

    for i = 2 : dDepth
        data = DataIn0(:,i:dDepth:end)./DataInAll;                         %% BECAREFUL WITH THIS APPROXIMATION
                                                                           % the conditional probability 
        datab = data;
        datab(isnan(datab)) = 0;
        switch caseI{2}(1)
            case 1
                data = datab;  
            case 2
                [row,col] = find(isnan(data));
                col0 = unique(col);
                [gc,~] = groupcounts(col);
                val0 = sum(data,1,'omitnan')/yLength;
%                 val0 = nansum(data,1,'omitnan');
                val0 = repelem(val0(col0),gc);
                data(sub2ind(size(data),row,col)) = deal(val0);
            case 3
                data(isnan(data)) = 1;  
        end

        clear row col col0 gc val0
            

            
        mapf = cell(yLength,1);                                            % mapping (threshold) first, then average
        avgf = cell(yLength,1);                                            % average first, then mapping (threshold)
        avgfb = avgf;
        for j = 1: yLength

            db = repmat(datab(j,:),rLength,1);
            db = abs(db - db');
            avgfb{j} = db;

            d0 = repmat(data(j,:),rLength,1);
            d0 = abs(d0 - d0');
            avgf{j} = d0;

            d1 = zeros(size(d0));
            d1(find(d0 <= Sthre)) = 1;
            d1(find(d0 > Sthre)) = 0;                                      %% BECAREFUL WITH THIS APPROXIMATION
            d1 = d1 - diag(diag(d1));
            mapf{j} = d1;

        end
        clear d0 d1

        mapf = cell2mat(mapf);                                             %% applied threshold
        avgf = cell2mat(avgf);                                             %% no threshold
        datab = zeros(rLength);
        avgfb = cell2mat(avgfb);

        biasMR = datab;

        for ii = 1 : rLength                                               %% BECAREFUL WITH THIS APPROXIMATION             
            
            switch caseI{3}

                case 1
                datab(ii,:)  = mean(avgf(ii:rLength:end,:));               % avg(yk)
                case 2
                datab(ii,:) = trimmean(avgf(ii:rLength:end,:),20);         % trimmed mean avg(yk)                                
                case 3
                datab(ii,:) = avgf(end-rLength+ii,:);                      % most recent
                case 4
                datab(ii,:) = mean(mapf(ii:rLength:end,:));                % trimeed f(avg(|p1-p1|))
                case 5
                datab(ii,:) = trimmean(mapf(ii:rLength:end,:),20);         % trimmed mean avg(yk)                                
                case 6
                datab(ii,:) = mapf(end-rLength+ii,:);                      % most recent
        
            end
            biasMR(ii,:) = mean(avgfb(ii:rLength:end,:));  
            
        end
        clear ii

        if caseI{3} <= 2
            datab(find(datab <= Sthre)) = 1;  
            datab(find(datab > Sthre)) = 0;   
            datab = datab- diag(diag(datab));
        end

        BiasM01{i-1} = datab;
        BiasM{i-1} = biasMR;                                               %% BECAREFUL HERE
    end
    
    clear i
    
    BiasV = zeros(size(BiasM));
    BiasV = repmat(BiasV,2,1);

    for i = 1: size(BiasM,2)
        ind000 = find(BiasM{1,i}<= 0.1);
        BiasM02 = zeros(size(BiasM{i}));
        BiasM02(ind000 ) = deal(1); 


        BiasV(1,i) = (nnz(BiasM02)- size(BiasM02,1))/(numel(BiasM02) - size(BiasM02,1));  % for average, the number of >threshold
        BiasV(2,i) = sum(BiasM01{i},'all')/(numel(BiasM01{i}) -size(BiasM02,1));  % for mapping, the average                     %% the averaged percenatge of bias
    end
    clear BiasM02 
%% Coefficients
    c = caseI{4};                                                          %% BECAREFUL WITH THIS APPROXIMATION

    BiasM0 = c(1) + c(2).*plus(BiasM01{:}) + c(3).*times(BiasM01{:});
    BiasM = mean(BiasM0,'all');                                            %% which bias should be used for threshold

    if size(DataInAll,2) == 2
        BiasM = 1;
    end
%% Stopping criteria
    if BiasM == 1
        DataOut = [];
        IndexOut = [];
        BiasM = [];
        BiasV = [];

    else

%% 
        l0 = size(DataInAll,2);
        xl = sum([1:1:l0-1]);
        x0 = zeros(xl,1);

        ff = CorM.*lambda(1) + BiasM0.*lambda(2);
        ffi = tril(true(size(ff)),-1);
        f = -ff(ffi).';
    
        Aeq = zeros(l0,l0*(l0-1)/2);
        aa = 1;
        for i = 1:l0
            bb = l0-i;
            Aeq(i,aa:aa+bb-1) = ones(1,bb);
            Aeq(i+1:l0-1,aa:aa+bb-2)=eye(l0-i-1);
            aa = aa+bb;
        end  
        
        aa = 0;
        for i = 1:l0-2
            aa = aa + l0-i;
            Aeq(l0,aa) = 1;
        end                                                                % last row
    
    
        beq = ones(l0,1);
        lb = zeros(l0*(l0-1)/2,1);
        ub = ones(l0*(l0-1)/2,1);
        intcon = [1:l0*(l0-1)/2];
    
        options = optimoptions('intlinprog','Display','off');
        x = intlinprog(f,intcon,[],[],Aeq,beq,lb,ub,x0,options);

        
        XX = zeros(l0,l0);
        y = x;
    
        for i = 1:l0-1
            XX(i,i+1:l0) = y(1:l0-i)';
            y(1:l0-i)=[];
        end
    
        XX = XX + XX';
        [ind1,ind2] = find(reshape(XX,l0,l0)>0.5);  
        
        
        for i = 1:length(ind1)
            if ind2(i)~=0
                ind2(ind1(i)) = 0;
                ind1(ind1(i)) = 0;
            end
        end
        indd = [ind1,ind2];
        indd(find(indd(:,1)==0),:) = [];
        
        DataOut = cell(1,size(indd,1));
        DataOut0 = [];
        
        
        
        for i = 1:length(indd)
            for j = 1 : dDepth
                DataOut1 = [DataIn0(:,j + (indd(i,1)-1)*dDepth) + DataIn0(:,j + (indd(i,2)-1)*dDepth)];
                DataOut0 = [DataOut0,DataOut1];
            end
            DataOut{i} = DataOut0;
            DataOut0 = [];
        end
        
        
        IndexOut = indd;

    end



end