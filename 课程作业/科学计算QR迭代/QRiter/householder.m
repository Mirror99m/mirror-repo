function H=householder(x)
    e=zeros(length(x),1);
    e(1)=1;
    v=x-sqrt((x')*x)*e;
    H=eye(length(x))-2/((v')*v)*v*(v');
end