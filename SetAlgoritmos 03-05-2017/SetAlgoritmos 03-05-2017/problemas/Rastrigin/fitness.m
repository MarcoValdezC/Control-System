function f = fitness(x)
A=10;
[~,n]=size(x);

f =A*n;
for i=1:n
    f = f+(x(i)^2 - A*cos(2*pi*x(i)));
end

end
