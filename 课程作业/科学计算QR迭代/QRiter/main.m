function main()
    % 参数设置
    n = 5; % 矩阵大小
    maxIter = 100; % 最大迭代次数
    tol = 1e-6; % 收敛阈值

    % 随机生成实对称矩阵
    A = randn(n); % 随机矩阵
    A = (A + A') / 2; % 生成实对称矩阵

    % 显式QR迭代
    [eigvalsQR, iterQR] = explicitQRIteration(A, maxIter, tol);

    % 隐式QR迭代
    [eigvalsIQR, iterIQR] = implicitQRIteration(A, maxIter, tol);

    % 绘制迭代图像
    figure;
    subplot(1, 2, 1);
    plot(abs(diag(iterQR)));
    title('显式QR迭代 - 特征值收敛');
    xlabel('迭代次数');
    ylabel('特征值绝对值');
    grid on;

    subplot(1, 2, 2);
    plot(abs(diag(iterIQR)));
    title('隐式QR迭代 - 特征值收敛');
    xlabel('迭代次数');
    ylabel('特征值绝对值');
    grid on;

    % 输出结果
    disp('显式QR迭代的特征值:');
    disp(eigvalsQR);
    disp('隐式QR迭代的特征值:');
    disp(eigvalsIQR);
end


function [eigvals, iter] = explicitQRIteration(A, maxIter, tol)
    n = size(A, 1);
    iter = zeros(n, maxIter); % 保存每次迭代的对角线元素
    for k = 1:maxIter
        [Q, R] = qr(A); % QR分解
        A = R * Q; % 更新矩阵
        iter(:, k) = diag(A); % 保存对角线元素
        % 检查收敛
        if max(abs(diag(A, -1))) < tol
            break;
        end
    end
    eigvals = diag(A); % 特征值
end

function [eigvals, iter] = implicitQRIteration(A, maxIter, tol)
    n = size(A, 1);
    iter = zeros(n, maxIter); % 保存每次迭代的对角线元素
    % 将矩阵化为上Hessenberg矩阵
    [Q, H] = hessenberg(A);
    A = H;
    for k = 1:maxIter
        % 选择Wilkinson位移
        delta = 0.5 * (A(n-1,n-1) - A(n,n));
        mu = A(n,n) + delta - sign(delta) * sqrt(delta^2 + A(n-1,n)^2);
        % 隐式QR迭代
        H_shifted = A - mu * eye(n);
        [Q_iter, R] = qr(H_shifted);
        A = R * Q_iter + mu * eye(n);
        iter(:, k) = diag(A); % 保存对角线元素
        % 检查收敛
        if max(abs(diag(A, -1))) < tol
            break;
        end
    end
    eigvals = diag(A); % 特征值
end

function [Q, H] = hessenberg(A)
    n = size(A, 1);
    Q = eye(n);
    for k = 1:n-2
        x = A(k+1:end, k);
        e1 = zeros(length(x), 1);
        e1(1) = norm(x);
        v = x + sign(x(1)) * e1;
        v = v / norm(v);
        Q_k = eye(n);
        Q_k(k+1:end, k+1:end) = eye(n-k) - 2 * (v * v');
        Q = Q * Q_k;
        A = Q_k' * A * Q_k;
    end
    H = A;
end