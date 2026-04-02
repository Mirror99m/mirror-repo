function [c,s]=givens(x)
    r=sqrt((x')*x);
    c=x(1)/r;
    s=x(2)/r;
end