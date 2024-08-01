point_clouds_path = '/Users/julian/stroke_research/brain_and_donuts/point_clouds.mat';
point_clouds = load(point_clouds_path);
point_clouds = point_clouds.point_clouds;

distanceBoundOnEdges = 2;

H0PersistanceArray = {};
H1PersistanceArray = {};
n_edges_array = {};
n_columns_array = {};
for subj = 1:length(point_clouds)
    [H1, H0, n_edges, n_columns] = rca1pc(point_clouds{1, subj}, distanceBoundOnEdges);
    H0PersistanceArray{subj, 1} = H0;
    H1PersistanceArray{subj, 1} = H1;
    n_edges_array{subj, 1} = n_edges;
    n_columns_array{subj, 1} = n_columns;
end

save(strcat('H0Persistance_d', string(distanceBoundOnEdges),'.mat'),'H0PersistanceArray') 
save(strcat('H1Persistance_d', string(distanceBoundOnEdges),'.mat'),'H1PersistanceArray') 
save(strcat('EdgesCount_d', string(distanceBoundOnEdges),'.mat'),'n_edges_array') 
save(strcat('ColumnsCount_d', string(distanceBoundOnEdges),'.mat'),'n_columns_array') 



