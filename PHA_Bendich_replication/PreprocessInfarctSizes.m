function [mdata, vInfarctSizesWDatas, vSexWDatas, vrange, colmap, markerstr] = PreprocessInfarctSizes(dgmArray, InfarctSizes, Sexes, start, stop)
%PREPROCESSINFARCTSIZES Summary of this function goes here
%   Detailed explanation goes here


% preprocess data (same cases were removed upstream from our analysis, indicated by empty array elements)
index = zeros(1,length(dgmArray));
for i=1:length(dgmArray)
    if (isempty(dgmArray{i}))
        continue;
    else
        index(i) = 1;
    end
end
index = logical(index);
vCaseNumWData = dgmArray(index);
nCaseWD       = numel(vCaseNumWData);
vInfarctSizesWData     = (InfarctSizes(index));
vSexWData     = (Sexes(index));

% produce colormap for plotting
nfifth = ceil((nCaseWD - 1) / 5) ;
del = 1 / nfifth ;
vwt = (0:del:1)' ;
colmap = [flipud(vwt), zeros(nfifth+1,1), ones(nfifth+1,1)] ;
colmap = colmap(1:size(colmap,1)-1,:) ;
%  cutoff last row to avoid having it twice
colmap = [colmap; ...
    [zeros(nfifth+1,1), vwt, ones(nfifth+1,1)]] ;
colmap = colmap(1:size(colmap,1)-1,:) ;
%  cutoff last row to avoid having it twice
colmap = [colmap; ...
    [zeros(nfifth+1,1), ones(nfifth+1,1), flipud(vwt)]] ;
colmap = colmap(1:size(colmap,1)-1,:) ;
%  cutoff last row to avoid having it twice
colmap = [colmap; ...
    [vwt, ones(nfifth+1,1), zeros(nfifth+1,1)]] ;
colmap = colmap(1:size(colmap,1)-1,:) ;
%  cutoff last row to avoid having it twice
colmap = [colmap; ...
    [ones(nfifth+1,1)], flipud(vwt), zeros(nfifth+1,1)] ;

% set marker types for plotting
markerstr = [] ;
for i = 1:length(vSexWData) ;
    if vSexWData(i) == 1 ;
        markerstr = strvcat(markerstr,'o') ;
    elseif vSexWData(i) == 2 ;
        markerstr = strvcat(markerstr,'+') ;
    elseif vSexWData(i) == 3 ;
        markerstr = strvcat(markerstr,'*') ;
    else
        disp(['!!!   Error:  for i = ' num2str(i)]) ;
        disp('!!!   failed to assing symbol to sex   !!!') ;
    end ;
end ;

% sort according to age
[temp,visort] = sort(vInfarctSizesWData,'ascend') ;
vInfarctSizesWDatas = vInfarctSizesWData(visort)' ;
vSexWDatas = vSexWData(visort)' ;
markerstr = markerstr(visort) ;


% Summarize data to quantiles of top bar lengths
vrange = start:stop;
mdatar = [] ;
for  iCaseWD = 1:nCaseWD ;
    iCase = vCaseNumWData{iCaseWD} ;
    mbar = iCase(:,2)-iCase(:,1);
    vbarlen = mbar ;
    vbarlens = sort(vbarlen,'descend') ;
    if (length(vbarlens)<stop)
        temp = zeros(stop,1);
        temp(1:length(vbarlens),1) = vbarlens;
        vbarlens = temp;
    end
    vbarlens = vbarlens(vrange) ;
    mdatar = [mdatar vbarlens] ;
end ;

% sort feature vectors according to age
mdata = mdatar(:,visort') ;
mdata = log10(mdata) ;



end


