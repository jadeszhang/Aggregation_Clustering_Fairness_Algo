%% Main function for aggregation

%%% Inputs: 
% Tbl: Binarized datatable
% ind: Race index 
% BinM00: index for chosen features for aggregation
% Sthre: stopping criteria
% caseI: switching between different cases

%%% Outputs:
% ProbD: datasets (year * chosen features +1 )
% ProbD{i}(:,1): counts on race
% ProbD{i}(:,j): coutns on (race intersecting feature)
% dataF: counts on aggregated race
% indLabel: race label after each aggregation
% indInd: race index after each aggregation
% bias: averaged bias on each feature


%%% notes:
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
%% Data input
clear all
clc
% diary biasOutputSummary

Tbl = readtable('CleanedData1.csv','PreserveVariableNames',true);
VarNames = Tbl.Properties.VariableNames;  
Data = table2array(Tbl);
Data(isnan(Data))=0;

% Dataset info
% Binary settings:
% first = 1                              (col2)
% new = 1                                (col3)
% f = 1                                  (col4)
% CARes = 1,internaltional = 1           (col5)(col6)
 
ind = 8; 
% binary index 
RaceIndex = VarNames(ind:end);                                             % non-binary index
MaxM1 = [2,3,4,5;
         2,3,4,6;
         2,3,4,7];                                                         % non-eth attributes: 
                                                                           % For each three-attributes feature, then we have extra 2 rows

MaxM2 = [8:1:length(RaceIndex)+ind-1];
yNumber = max(Data(:,1))-min((Data(:,1)))+1;                               % # of years
yLength = size(Data,1)/yNumber; 
BinM00 = {[0,4,2; 1,1,1;...
                           ],[3]};                                         % Aggregation at the layer i with BinM00{1,1}(i,:) == 0
% change your parameters here
Sthre = 0.1;
% similarity ; fairness
lambda = [0.0,1.0];
caseI = {2,[1,2],5,[1/3, 1/3, 1/3]};




%% Create time series data

ProbD = cell(1,length(RaceIndex));



for i = 1:length(RaceIndex)
    data = Data(:,[1:ind-1, ind-1+i]);
    data1 = zeros(yNumber,size(BinM00{1,1},2));

    for yi = 1:yNumber
        data0 = data((yi-1)*yLength+1:yi*yLength,:);
        total0 = sum(data0(:,end));
        if total0 == 0
           data1(yi,:) = deal(0);
        else
            for j = 1:size(BinM00{1,1},2)
                
                jind = BinM00{1,1}(1,j);
                if jind == 0
                    data1(yi,j) = sum(data0(:,end),'all');
                else
                    ind0 = find(data0(:,jind)==1);
                    data1(yi,j) =  sum(data0(ind0,end),'all');
                end
            end
        end
    end
    ProbD(:,i) = {data1};

end
clear i

% Standardization
ProbD = cell2mat(ProbD);
ProbD0 = ProbD(:,1:BinM00{end}:end);
ProbD0 = sum(ProbD0,2);
ProbD1 = ones(size(ProbD0))*max(ProbD0);
ProbD0 = ProbD1./ProbD0;
ProbD = ProbD.*ProbD0;

clear PorbD0 ProbD1
ProbD = mat2cell(ProbD,yNumber,ones(1,size(RaceIndex,2))*BinM00{end});
%% aggregation per each case

VarNamesR = VarNames(ind:end);

currentFolder = pwd;
folder0 = strcat(currentFolder,'\Outputs');





[dataF,indLabel,indInd,bias] = psedoAggre(ProbD,Sthre,VarNamesR,lambda,caseI);



