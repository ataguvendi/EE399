function [reference_x, reference_y,fevalcount,gradcount] = linsearch(startx, stepsize, maxit, fn, fngrad)
it = 0;
g = fngrad(fn,startx);
gradcount = 1;

reference_x = startx;
reference_y = fn(startx);
fevalcount = 1;

temp = startx;
while it<maxit
temp = temp-(stepsize.*g);
tempf = fn(temp);
fevalcount = fevalcount+1;

if tempf<reference_y
reference_x = temp;
reference_y = tempf;

end


it = it+1;
end



end





