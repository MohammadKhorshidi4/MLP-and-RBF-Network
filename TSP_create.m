function pop = TSP_create(nvars,FitnessFcn,options)

PopulationSize = options.PopulationSize ;

pop = zeros(PopulationSize,nvars) ;
for i=1:PopulationSize
    pop(i,:) = randperm(100,nvars) ;
end
