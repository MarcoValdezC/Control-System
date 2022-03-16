function res=zdt2(x)
	g = 0;
    f1 = x(1);
    for i = 2:30
        g = g + x(i);
    end
    
	g = 1 + 9 * g / (30 - 1);
	h = 1 - (f1/g)^2;

	res = [f1 ; g*h];
   