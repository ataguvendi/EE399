function prime = numderivxd(ff,x)
%numbderiv In place derivative using limit definition.
%Uses fn3d definition of function. Valid for all dimensions.
h=0.001;
prime = NaN(size(x));

for i = 1:length(x)
    plush = zeros(size(x));
    plush(i) = h;
    prime(i) = ((ff(x+plush))-ff(x))./h;
end
end
