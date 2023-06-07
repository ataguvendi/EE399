close all;
clear;
clc;
searchradii = [0.1,1,10];
lr=(1:1.5:10).*0.01;
drivfn(@ackley, @numderivxd, 0, searchradii,2, lr, "Ackley 2D",100);
drivfn(@ackley, @numderivxd, 0, searchradii,3, lr, "Ackley 3D",100);
drivfn(@hart6, @numderivxd, 0, searchradii,6, lr, "Hartmann 6D",10);
drivfn(@bukin6, @numderivxd, 0, searchradii,2, lr, "Bukin 6, 2D",100);
drivfn(@levy, @numderivxd, 0, searchradii,2, lr, "Levy 2D",10);
drivfn(@levy, @numderivxd, 0, searchradii,3, lr, "Levy 3D",10);

function drivfn(ff,fgrad, center, searchradii, dim, lr,funcdesription,numsim)
%driver - runs the functions.


runcount = numsim;  %times to run each function at every (searchradius,lr)

%Preallocations for final results reporting, runtimes and cost = abs(Z)
accross_search_runtimes = NaN(length(lr),length(searchradii));
accross_search_runtimes_m = accross_search_runtimes;
accross_search_runtimes_gd = accross_search_runtimes;
accross_search_runtimes_ls = accross_search_runtimes;
accross_search_runtimes_sa = accross_search_runtimes;
accross_search_runtimes_chain = accross_search_runtimes;

accross_search_feval = accross_search_runtimes;
accross_search_feval_m = accross_search_runtimes;
accross_search_feval_gd = accross_search_runtimes;
accross_search_feval_ls = accross_search_runtimes;
accross_search_feval_sa = accross_search_runtimes;
accross_search_feval_chain = accross_search_runtimes;

accross_search_geval = accross_search_runtimes;
accross_search_geval_m = accross_search_runtimes;
accross_search_geval_gd = accross_search_runtimes;
accross_search_geval_ls = accross_search_runtimes;
accross_search_geval_sa = accross_search_runtimes;
accross_search_geval_chain = accross_search_runtimes;


accross_search_deviations = accross_search_runtimes;
accross_search_deviations_m = accross_search_runtimes;
accross_search_deviations_gd = accross_search_runtimes;
accross_search_deviations_ls = accross_search_runtimes;
accross_search_deviations_sa = accross_search_runtimes;
accross_search_deviations_chain = accross_search_runtimes;

for current_learning_rate_index = 1:length(lr)
for search_radius_index = 1:length(searchradii)
    %Individual (searchradius,lr) trackers, will be averaged for final
    %reporting
    searchradius = searchradii(search_radius_index);
    runtimes = NaN(runcount,1);
    runtimes_chain = runtimes;
    runtimes_mtlb = runtimes;
    runtimes_gd = runtimes;
    runtimes_ls = runtimes;
    runtimes_sa = runtimes;

    divergence = NaN(runcount,1);
    divergence_chain = divergence;
    divergence_mtlb = divergence;
    divergence_gd = divergence;
    divergence_ls = divergence;
    divergence_sa = divergence;

    feval = NaN(runcount,1);
    feval_chain = divergence;
    feval_mtlb = divergence;
    feval_gd = divergence;
    feval_ls = divergence;
    feval_sa = divergence;

    geval = NaN(runcount,1);
    geval_chain = divergence;
    geval_mtlb = divergence;
    geval_gd = divergence;
    geval_ls = divergence;
    geval_sa = divergence;

    %start at random point arounf the center
    starts = center + rand(runcount,dim).*2.*searchradius - searchradius;
    
    %Hyperparameter setup for decaying learning rate. Will break at the
    %learning rate
    break_cond = lr(current_learning_rate_index);
    start_rate = 10e7*break_cond;%set to a high number to begin decay schedule at.



    %set up matlab optimixation toolbox, lower and upper bounds.
    lb = ones(1,dim) .* (-searchradius) + center;
    ub = ones(1,dim) .* (searchradius) + center;

    %Setup maximum number of iterations for linesearch and gradient descent
    maxit = 10e3;
   

    for iteration = 1:runcount
        %chained algorithm
        tic
        [~,output,a,b] = chainlink(starts(iteration,:),start_rate, break_cond, searchradius,center,ff, fgrad, break_cond^2, maxit);
        runtimes_chain(iteration) = toc;
        divergence_chain(iteration) = abs(output);
        feval_chain(iteration)=a;
        geval_chain(iteration)=b;


        %custom decay algorithm
        tic
        [~,output,a,b] = decayxd(starts(iteration,:),start_rate, break_cond, searchradius,0,ff, fgrad,maxit);
        runtimes(iteration) = toc;
        divergence(iteration) = abs(output);
        feval(iteration)=a;
        geval(iteration)=b;


        %matlab
        tic;
        [~,fval]= fmincon(ff,starts(iteration,:),[], [], [], [], lb, ub);

        runtimes_mtlb(iteration) = toc;
        divergence_mtlb(iteration) = abs(fval);
        feval_mtlb(iteration)=NaN;
        geval_mtlb(iteration)=NaN;
        %standard gd
        tic;
        [~,fval,a,b] = graddesc(starts(iteration,:), break_cond, maxit, ff, fgrad);
        runtimes_gd(iteration) = toc;
        divergence_gd(iteration)=abs(fval);
        feval_gd(iteration)=a;
        geval_gd(iteration)=b;

        %backtrack linsearch
        tic;
        [~,fval,a,b] = linsearch(starts(iteration,:), break_cond, maxit, ff, fgrad);
        runtimes_ls(iteration) = toc;
        divergence_ls(iteration) = abs(fval);
        feval_ls(iteration)=a;
        geval_ls(iteration)=b;
        %simulated annealing
        tic;
        [~,fval,a,b] = sa(starts(iteration,:), 1, 0.99, 0.01, maxit, 0.01, ff);
        runtimes_sa(iteration) = toc;
        divergence_sa(iteration) = abs(fval);
        feval_sa(iteration)=a;
        geval_sa(iteration)=b;
    end
    hold off;
    
    %Set the row and column that corresponds to the learning rate as 
    accross_search_deviations(current_learning_rate_index,search_radius_index) = mean(divergence);
    accross_search_deviations_m(current_learning_rate_index,search_radius_index) = mean(divergence_mtlb);
    accross_search_deviations_gd(current_learning_rate_index,search_radius_index) = mean(divergence_gd);
    accross_search_deviations_ls(current_learning_rate_index,search_radius_index) = mean(divergence_ls);
    accross_search_deviations_sa(current_learning_rate_index,search_radius_index) = mean(divergence_sa);
    accross_search_deviations_chain(current_learning_rate_index,search_radius_index) = mean(divergence_chain);
 
    accross_search_runtimes(current_learning_rate_index,search_radius_index) = mean(runtimes);
    accross_search_runtimes_m(current_learning_rate_index,search_radius_index) = mean(runtimes_mtlb);
    accross_search_runtimes_gd(current_learning_rate_index,search_radius_index) = mean(runtimes_gd);
    accross_search_runtimes_ls(current_learning_rate_index,search_radius_index) = mean(runtimes_ls);
    accross_search_runtimes_sa(current_learning_rate_index,search_radius_index) = mean(runtimes_sa);
    accross_search_runtimes_chain(current_learning_rate_index,search_radius_index) = mean(runtimes_chain);    

    accross_search_feval(current_learning_rate_index,search_radius_index) = mean(feval);
    accross_search_feval_m(current_learning_rate_index,search_radius_index) = mean(feval_mtlb);
    accross_search_feval_gd(current_learning_rate_index,search_radius_index) = mean(feval_gd);
    accross_search_feval_ls(current_learning_rate_index,search_radius_index) = mean(feval_ls);
    accross_search_feval_sa(current_learning_rate_index,search_radius_index) = mean(feval_sa);
    accross_search_feval_chain(current_learning_rate_index,search_radius_index) = mean(feval_chain); 

    accross_search_geval(current_learning_rate_index,search_radius_index) = mean(geval);
    accross_search_geval_m(current_learning_rate_index,search_radius_index) = mean(geval_mtlb);
    accross_search_geval_gd(current_learning_rate_index,search_radius_index) = mean(geval_gd);
    accross_search_geval_ls(current_learning_rate_index,search_radius_index) = mean(geval_ls);
    accross_search_geval_sa(current_learning_rate_index,search_radius_index) = mean(geval_sa);
    accross_search_geval_chain(current_learning_rate_index,search_radius_index) = mean(geval_chain); 

end
end
figure;
hold on;
alpha = 0.65;
mesh(searchradii, lr, accross_search_deviations,'FaceColor',[0 0.4470 0.7410], 'FaceAlpha', alpha, "EdgeAlpha",alpha); %blue
mesh(searchradii, lr, accross_search_deviations_chain, 'FaceColor',[0.4660 0.6740 0.1880], 'FaceAlpha', alpha, "EdgeAlpha", alpha); %green
mesh(searchradii, lr, accross_search_deviations_m,'FaceColor',[0.6350 0.0780 0.1840],'FaceAlpha', alpha, "EdgeAlpha", alpha); %red
mesh(searchradii, lr, accross_search_deviations_gd, 'FaceColor',[0.9290 0.6940 0.1250],'FaceAlpha', alpha, "EdgeAlpha", alpha); %gold
mesh(searchradii, lr, accross_search_deviations_ls, 'FaceColor','#E3B5A4', 'FaceAlpha', alpha, "EdgeAlpha", alpha); %turqouise
mesh(searchradii, lr, accross_search_deviations_sa, 'FaceColor','#000000', 'FaceAlpha', alpha, "EdgeAlpha", alpha);%black
view(3);
title("Costs of Algorithms " + funcdesription);
xlabel("Search radius");
ylabel("Learning rate")
zlabel("Cost");
set(gca, 'XScale', 'log');
set(gca, 'ZScale', 'log');
legend("Decaying Learning Rate with Pivots","Chained Implementation", "Matlab Optimization", "Gradient Descent", "Linesearch", "SA");
hold off;

figure;
hold on;
alpha = 0.65;
mesh(searchradii, lr, accross_search_runtimes,'FaceColor',[0 0.4470 0.7410], 'FaceAlpha', alpha, "EdgeAlpha",alpha); %blue
mesh(searchradii, lr, accross_search_runtimes_chain, 'FaceColor',[0.4660 0.6740 0.1880], 'FaceAlpha', alpha, "EdgeAlpha", alpha); %green
mesh(searchradii, lr, accross_search_runtimes_m,'FaceColor',[0.6350 0.0780 0.1840],'FaceAlpha', alpha, "EdgeAlpha", alpha); %red
mesh(searchradii, lr, accross_search_runtimes_gd, 'FaceColor',[0.9290 0.6940 0.1250],'FaceAlpha', alpha, "EdgeAlpha", alpha); %gold
mesh(searchradii, lr, accross_search_runtimes_ls, 'FaceColor','#E3B5A4', 'FaceAlpha', alpha, "EdgeAlpha", alpha); %turqouise
mesh(searchradii, lr, accross_search_runtimes_sa, 'FaceColor','#000000', 'FaceAlpha', alpha, "EdgeAlpha", alpha);%black
view(3);
title("Runtimes of Algorithms " + funcdesription);
xlabel("Search radius");
ylabel("Learning rate")
zlabel("Runtime");
set(gca, 'XScale', 'log');
set(gca, 'ZScale', 'log');
legend("Decaying Learning Rate with Pivots","Chained Implementation", "Matlab Optimization", "Gradient Descent", "Linesearch", "SA");
hold off;


figure;
hold on;
alpha = 0.65;
mesh(searchradii, lr, accross_search_feval,'FaceColor',[0 0.4470 0.7410], 'FaceAlpha', alpha, "EdgeAlpha",alpha); %blue
mesh(searchradii, lr, accross_search_feval_chain, 'FaceColor',[0.4660 0.6740 0.1880], 'FaceAlpha', alpha, "EdgeAlpha", alpha); %green
mesh(searchradii, lr, accross_search_feval_m,'FaceColor',[0.6350 0.0780 0.1840],'FaceAlpha', alpha, "EdgeAlpha", alpha); %red
mesh(searchradii, lr, accross_search_feval_gd, 'FaceColor',[0.9290 0.6940 0.1250],'FaceAlpha', alpha, "EdgeAlpha", alpha); %gold
mesh(searchradii, lr, accross_search_feval_ls, 'FaceColor','#E3B5A4', 'FaceAlpha', alpha, "EdgeAlpha", alpha); %turqouise
mesh(searchradii, lr, accross_search_feval_sa, 'FaceColor','#000000', 'FaceAlpha', alpha, "EdgeAlpha", alpha);%black
view(3);
title("#Function Evaluations of Algorithms " + funcdesription);
xlabel("Search radius");
ylabel("Learning rate")
zlabel("#Evals");
set(gca, 'XScale', 'log');
%set(gca, 'ZScale', 'log');
legend("Decaying Learning Rate with Pivots","Chained Implementation", "Matlab Optimization", "Gradient Descent", "Linesearch", "SA");
hold off;

figure;
hold on;
alpha = 0.65;
mesh(searchradii, lr, accross_search_geval,'FaceColor',[0 0.4470 0.7410], 'FaceAlpha', alpha, "EdgeAlpha",alpha); %blue
mesh(searchradii, lr, accross_search_geval_chain, 'FaceColor',[0.4660 0.6740 0.1880], 'FaceAlpha', alpha, "EdgeAlpha", alpha); %green
mesh(searchradii, lr, accross_search_geval_m,'FaceColor',[0.6350 0.0780 0.1840],'FaceAlpha', alpha, "EdgeAlpha", alpha); %red
mesh(searchradii, lr, accross_search_geval_gd, 'FaceColor',[0.9290 0.6940 0.1250],'FaceAlpha', alpha, "EdgeAlpha", alpha); %gold
mesh(searchradii, lr, accross_search_geval_ls, 'FaceColor','#E3B5A4', 'FaceAlpha', alpha, "EdgeAlpha", alpha); %turqouise
mesh(searchradii, lr, accross_search_geval_sa, 'FaceColor','#000000', 'FaceAlpha', alpha, "EdgeAlpha", alpha);%black
view(3);
title("#Gradient Evaluations of Algorithms " + funcdesription);
xlabel("Search radius");
ylabel("Learning rate")
zlabel("#Evals");
set(gca, 'XScale', 'log');
%set(gca, 'ZScale', 'log');
legend("Decaying Learning Rate with Pivots","Chained Implementation", "Matlab Optimization", "Gradient Descent", "Linesearch", "SA");
hold off;

end