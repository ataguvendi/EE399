runs = 100;

ackley_grad_performance = NaN(runs,1);
hart_grad_performance = ackley_grad_performance;
levy_grad_performance = ackley_grad_performance;

ackley_eval_performance = ackley_grad_performance;
hart_eval_performance = ackley_grad_performance;
levy_eval_performance = ackley_grad_performance;


for i = 1:runs
    point_ackley = rand(1,2);
    tic;
    ackley(point_ackley);
    ackley_eval_performance(i) = toc;
    tic;
    numderivxd(@ackley, point_ackley);
    ackley_grad_performance(i) = toc;

    point_hart = rand(1,6);
    tic;
    hart6(point_hart);
    hart_eval_performance(i) = toc;
    tic;
    numderivxd(@hart6, point_hart);
    hart_grad_performance(i) = toc;

    point_levy = rand(1,2);
    tic;
    levy(point_levy);
    levy_eval_performance(i) = toc;
    tic;
    numderivxd(@levy, point_levy);
    levy_grad_performance(i) = toc;
end
X = categorical({'Ackley','Hartmann','Levy'});
X = reordercats(X,{'Ackley','Hartmann','Levy'});

Y = [mean(ackley_eval_performance), mean(ackley_grad_performance); mean(hart_eval_performance), mean(hart_grad_performance); mean(levy_eval_performance), mean(levy_grad_performance)];

b = bar(X,Y,"grouped");
legend([b(1),b(2)],{'Function Evaluation','Gradient Evaluation'});
title("Function vs. Gradient Eval Times");
