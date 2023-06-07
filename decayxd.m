function [reference_x, reference_y,fevalcount,gradcount] = decayxd(startx, learn_rate_start, learn_rate_break, searchsize, searchcenter,fn,fngrad,maxit)
%tic
%Statistical Gradient Descent Optimization Algorithm
learn_rate = learn_rate_start;


reference_x = startx; %current point
reference_y = fn(reference_x);
fevalcount = 1;
gradcount = 0;
iteration_counter = 0;
points_to_gen = 30;
while iteration_counter <maxit
    iteration_counter = iteration_counter + 1;
    if learn_rate < learn_rate_break
        reference_y = fn(reference_x);
        fevalcount = fevalcount + 1;
        break;
    end
  
    thirthy_points = rand(points_to_gen,length(startx)).* (2.*searchsize) - searchsize + searchcenter; %generate pivot points in search window
    found = false;
    for i = 1:points_to_gen
        yval = fn(thirthy_points(i,:));
        fevalcount = fevalcount + 1;
        if yval<reference_y
            found = true;
            reference_y = yval;
            reference_x = thirthy_points(i,:);
        end
    end
    if found
        continue; %after pivot, cycle back, a new random point may prove more productive than a gradient step.
    end

    %apply standard gradient descent
    g = fngrad(fn,reference_x);
    gradcount = gradcount + 1;
    %g=fn(reference_x);
    temp_reference_x = reference_x-(learn_rate.*g); 
    temp_reference_y = fn(temp_reference_x);
    fevalcount = fevalcount+1;

    if temp_reference_y>reference_y %if no progress is made, decay the learning rate (monotonically decreasing check)
        learn_rate = learn_rate/10;
        continue;
    end
    if max(temp_reference_x > (searchsize+searchcenter)) ||  max(temp_reference_x < (searchsize+searchcenter)) %if out of bounds, reject step
        continue;
    end
    reference_x = temp_reference_x;
    reference_y = temp_reference_y;
    
end
end
