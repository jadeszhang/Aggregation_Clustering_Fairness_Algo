%% Function for psedoAggregation

%%% Inputs:
% DataIn: Binarized data
% Sthre: stopping criteria
% VarNamesR: varaible labels for race
% caseI: switch between different cases

%%% Outputs:
% dataOut: aggregated counts
% indLabelOut: race label after each aggregation
% indIndOut: race index after each aggregation
% biasOut: averaged bias on each feature

%%% notes:
% variable+num: intermediate level variables (ex. DataIn0, DataIn1...)



function [dataOut,indLabelOut,indIndOut,biasOut] = psedoAggre(DataIn,Sthre,VarNamesR,lambda,caseI)
    
    dDepth = size(DataIn{1,1},2);
    DataIn0 = cell2mat(DataIn); 
    Data1 = DataIn0(:,1:dDepth:end);
    clear DataIn0  
    indLabel = {VarNamesR};
    indInd = {string([1:1:size(Data1,2)])};

    biasOut = [];
    biasS = 0;
    Data1 = {Data1};
    DataIn0 = {DataIn};

    while size(indLabel{end},2) > 2  && biasS ~= 1

        DataIn = DataIn0{end};%
        Data1 = Data1{end};
        if mod(size(Data1,2),2) == 0

            [data00,ind00,biasS,biasV] = bigraph(DataIn,lambda,Sthre,caseI);
            data01 = cell2mat(data00);
            data01 = data01(:,1:dDepth:end);

            inddL = strings(size(ind00,1),1);
            inddI = strings(size(ind00,1),1);

            for i = 1:size(ind00)
                inddI(i) = strcat(indInd{1,end}(:,ind00(i,1))', ',',indInd{1,end}(:,ind00(i,2))');            
                inddL(i) = strcat(indLabel{1,end}(:,ind00(i,1))', ',',indLabel{1,end}(:,ind00(i,2))');            
            end

            indLabel = [indLabel, {inddL'}];
            indInd = [indInd, {inddI'}];
            biasOut = [biasOut,biasV];
            DataIn0 = [DataIn0, {data00}];
            Data1 = [Data1, {data01}];

        else
            data1 = sum(Data1,1);
            minInd = find(data1 == min(data1),1,'first');
            data1 = DataIn(:,minInd);
            DataIn(:,minInd) = [];
            [data00,ind00,biasS,biasV] = bigraph(DataIn,lambda,Sthre,caseI);
            data01 = cell2mat(data00);

            data02 = data01(:,1:dDepth:end);
            singleIn0 = find(sum(data02,1)== min(sum(data02,1)),1,'first');% add the singlet to the smallest pairs
            data02 = data01(:,(singleIn0-1)*dDepth+1:(singleIn0)*dDepth);
            data01(:,(singleIn0-1)*dDepth+1:(singleIn0)*dDepth) = data02 + cell2mat(data1);

            inddL = strings(size(ind00,1),1);
            inddI = strings(size(ind00,1),1);

            for i = 1:size(ind00)
                inddI(i) = strcat(indInd{1,end}(:,ind00(i,1))', ',',indInd{1,end}(:,ind00(i,2))');            
                inddL(i) = strcat(indLabel{1,end}(:,ind00(i,1))', ',', indLabel{1,end}(:,ind00(i,2))');
            end

            inddL(singleIn0,:) = strcat(inddL(singleIn0,:),',',num2str(indLabel{1,end}{minInd}));
            inddI(singleIn0,:) = strcat(inddI(singleIn0,:),',',num2str(minInd));

            indLabel = [indLabel, {inddL'}];
            indInd = [indInd, {inddI'}];
            biasOut = [biasOut,biasV];
            DataIn0 = [DataIn0, {data00}];
            Data1 = [Data1, {data01}];
        end

    end   

    dataOut = DataIn0;
    indLabelOut = indLabel;
    indIndOut = indInd;

end