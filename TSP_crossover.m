function Kids = TSP_crossover(parents,options,nvars,FitnessFcn,thisScore,thisPopulation)
M = ceil(length(parents)/2)  ;
Kids = zeros(M,nvars) ;
for i=1:M
    p1 = thisPopulation(parents(2*i-1),:) ;
    p2 = thisPopulation(parents(2*i),:) ;
    
    c = sort(ceil(length(p1)*rand(1,2))) ;
    
    Kid1 = p1 ;
    SelMat1 = p1(c(1):c(2)) ;
    
    [tf,loc] = ismember(SelMat1,p2) ;
    loc1 = loc(find(loc~=0));
    if length(c(1):c(2))==length(loc1)
        Kid1(c(1):c(2)) = p2(sort(loc1)) ;
    end
    Kids(i,:) = Kid1 ;
    
end
