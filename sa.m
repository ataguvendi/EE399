function [reference_x, reference_y,fevalcount,gradcount] = sa(startx, beta,initial_temp, terminal_temp, maxit, alteration_step_size, fn)
fevalcount = 1;
gradcount = 0;
reference_y = fn(startx);
reference_x = startx;
temperature = initial_temp;
iterations = maxit/10;
totalit = 0;

while temperature > terminal_temp && totalit < maxit
    totalit = totalit + 10;

    for i = 1:iterations
       %create neighbor solution, +- stepsize to random parameter.
      alteration_index = randi([0,1],size(reference_x));
      signchange = randi([1,2],size(reference_x));
      alteration = ((-1).^signchange) .* alteration_step_size.*alteration_index;
       %alteration = (-alteration_step_size) .* fngrad(fn,reference_x);
      

      %evaluate neighbor solution
      tempx = reference_x + alteration;
      tempy = fn(tempx);
      fevalcount = fevalcount + 1;

      
      if tempy<reference_y || ( rand() < exp( (tempy - reference_y) /temperature)  )% pivot
          reference_x = tempx;
          reference_y = tempy;
      end

    end

    temperature = temperature/(1+beta*temperature);
end







end