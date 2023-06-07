function [reference_x, reference_y,fevalcount,gradcount] = graddesc(startx, learn_rate, maxit, fn, fngrad)
    it = 0;
    reference_x = startx;
    gradcount = 0;
    stepsize = 1;
    while it<maxit && all(~(stepsize == 0))
        n_reference_x = reference_x - (learn_rate .* fngrad(fn,reference_x));
        gradcount = gradcount+1;
        it = it+1;
        stepsize = n_reference_x - reference_x;
        reference_x = n_reference_x;
        
    
    end 
    reference_y = fn(reference_x);
    fevalcount = 1;

end