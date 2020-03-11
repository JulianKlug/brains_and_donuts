function [rho, pval] = InfarctSizeAnalysis(dgmArray, InfarctSizes, Sexes, start, stop)
%INFARCTSIZEANALYSIS Summary of this function goes here
%   Detailed explanation goes here

% Preprocess data
[ mdata, vInfarctSizesWDatas, vSexWDatas, vrange, colmap, markerstr ] = PreprocessInfarctSizes(dgmArray, InfarctSizes, Sexes, start, stop);


%  Make PC1 vs. PC2 Scatterplot
figure;
paramstruct = struct('npc',2, ...
                     'iscreenwrite',1, ...
                     'viout',[0 0 0 0 1]) ;
outstruct = pcaSM(mdata,paramstruct) ;
mpc = getfield(outstruct,'mpc') ;
paramstruct = struct('icolor',2, ...
                     'markerstr',markerstr, ...
                     'xlabelstr','PC 1 Scores', ...
                     'ylabelstr','PC 2 Scores', ...
                     'labelfontsize',12, ...
                     'iscreenwrite',1) ;
projplot2SM(mpc,eye(2),paramstruct) ;


%  Make PC1 vs. infarct size Scatterplot
figure;
paramstruct = struct('npc',1,...
                     'iscreenwrite',1,...
                     'viout',[0 0 0 0 1]) ;
outstruct = pcaSM(mdata,paramstruct) ;
vscores = getfield(outstruct,'mpc') ;

hold on ;
for  iCaseWD = 1:length(mdata(1,:)) ;
  plot(vInfarctSizesWDatas(iCaseWD),vscores(iCaseWD),markerstr(iCaseWD), ...
                'Color',colmap(iCaseWD,:)) ;

end ;
vax = axisSM(vInfarctSizesWDatas,vscores) ;
axis(vax) ;

[rho,pval] = corr(vInfarctSizesWDatas',vscores','type','Pearson') ;
text(vax(1) + 0.05 * (vax(2) - vax(1)), ...
     vax(3) + 0.94 * (vax(4) - vax(3)), ...
     ['Pearson Correlation = ' num2str(rho)], ...
     'FontSize',12) ;
text(vax(1) + 0.05 * (vax(2) - vax(1)), ...
     vax(3) + 0.88 * (vax(4) - vax(3)), ...
     ['p-val = ' num2str(pval)], ...
     'FontSize',12) ;

xlabel('Infarct Size (voxels)','FontSize',12) ;
ylabel('PC1 Scores','FontSize',12) ;
hold off ;


end