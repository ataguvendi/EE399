function [finx,finy,fevalcount,gradcount] = chainlink(startx, learn_rate_start, learn_rate_break, searchsize, searchcenter, fn, fngrad, mingain, maxit)
%CHAINLINK Is a decaying search-box implementation of decayxd

arguments
startx %input for start place
learn_rate_start %highest learning rate
learn_rate_break %lowest learning rate
searchsize %searchsize
searchcenter %centered at
fn %target func handle
fngrad %gradient func handle
mingain %minimum gain to keep closing search window.
maxit %maximum iterations
end

[x,y,fevalcount,gradcount] = decayxd(startx, learn_rate_start, learn_rate_break, searchsize, searchcenter, fn, fngrad,maxit);
gain = y;
newx = x;
newy = y;

while gain > mingain %break condition
y=newy;
x=newx;
searchsize = searchsize/10;
searchcenter = x;
[newx, newy,a,b] = decayxd(x, learn_rate_start, learn_rate_break, searchsize, searchcenter, fn, fngrad,maxit);
fevalcount = fevalcount+a;
gradcount = gradcount+b;
gain = y-newy;
end

finx = x;
finy = min([newy,y]);

end