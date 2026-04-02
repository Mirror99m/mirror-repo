function H=hessenberg(A)
    H=A;
    [n,~]=size(A);
    for i=1:n-2
        Q1=householder(H(i+1:n,i));
        Q2=blkdiag(eye(i),Q1);
        H=Q2*H*(Q2');
    end
end