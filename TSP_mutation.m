function Kids = TSP_mutation(parents,options,nvars,FitnessFcn, state, thisScore,thisPopulation)
Kids = zeros(length(parents),nvars) ;
for i=1:length(parents)
    c = ceil(nvars*rand(1,2)) ;
    parent = thisPopulation(parents(i),:);
    newKid = parent ;
    newKid(c(1)) = parent(c(2)) ;
    newKid(c(2)) = parent(c(1)) ;
    
    Kids(i,:) = newKid ;
end
