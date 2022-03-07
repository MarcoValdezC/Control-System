function f = fitness(x)
[~,n]=size(x);

f =0;
for i=1:n
    f=f+x(i)^2;
end
end
