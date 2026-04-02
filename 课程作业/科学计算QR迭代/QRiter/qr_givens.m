function A=qr_givens(H)%输入一个上hessenberg矩阵，用givens进行合同变换，输出仍为上hessenberg矩阵
    A=H;
    sigma=A(end,end);
    [n,~]=size(A);
    for i=1:n-1
        %[c,s]=givens(A([i,i+1],i));%不位移
        [c,s]=givens(A([i,i+1],i)-[sigma;0]);%位移A(n,n)
        G=blkdiag(eye(i-1),[c,s;-s,c],eye(n-i-1));
        A=G*A*(G');
    end
end