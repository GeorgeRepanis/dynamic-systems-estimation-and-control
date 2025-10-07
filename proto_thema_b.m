% Adaptive Estimation with Bias (Polarization Error) - Final Version

clear; clc; close all;

%% System Parameters
A_true = [-2 0.25; -0.75 -2];
B_true = [0; 1.5];

%% Simulation Settings
dt = 0.01;
t_end = 20;
t = 0:dt:t_end;
N = length(t);
n = 2; m = 1;

%% Input Signal
u = sin(2*t) + 0.5*sin(5*t);

%% Polarization Error (bounded)
omega_bar = 0.2;               % Max norm
omega = omega_bar * [sin(0.5*t); cos(0.5*t)];

%% Initial Conditions
x = zeros(n, N);
x_hat = zeros(n, N);
e_x = zeros(n, N);
x(:,1) = [0.5; 0];
x_hat(:,1) = [0; 0];

%% Parameter Estimation Initialization
theta = zeros(n*(n+m), N);
theta(:,1) = 0.1*randn(n*(n+m),1);
Gamma = 10*eye(n*(n+m));

%% Simulation Loop
for k = 1:N-1
    % True system dynamics with bias
    dx = A_true*x(:,k) + B_true*u(k) + omega(:,k);
    x(:,k+1) = x(:,k) + dt*dx;

    % Observer dynamics
    A_hat = reshape(theta(1:n*n,k),n,n);
    B_hat = reshape(theta(n*n+1:end,k),n,m);
    dx_hat = A_hat*x_hat(:,k) + B_hat*u(k);
    x_hat(:,k+1) = x_hat(:,k) + dt*dx_hat;

    % Estimation error
    e = x(:,k) - x_hat(:,k);
    e_x(:,k) = e;

    % Regression vector
    phi = [kron(x_hat(:,k)',eye(n)), u(k)*eye(n)];

    % Parameter update law
    dtheta = Gamma * phi' * e;
    theta(:,k+1) = theta(:,k) + dt*dtheta;
end

%% Final parameter matrices
A_hat_final = reshape(theta(1:n*n,end),n,n);
B_hat_final = reshape(theta(n*n+1:end,end),n,m);

%% Extracting parameter traces
A_hat_all = theta(1:n*n,:);
B_hat_all = theta(n*n+1:end,:);

%% Plotting
% Figure 1: u(t), x1, x2
figure;
subplot(3,1,1); plot(t,u,'m'); title('u(t) - πολυαρμονική'); grid on;
subplot(3,1,2); plot(t,x(1,:),'b',t,x_hat(1,:),'r--'); legend('x_1','\hat{x}_1'); grid on;
subplot(3,1,3); plot(t,x(2,:),'b',t,x_hat(2,:),'r--'); legend('x_2','\hat{x}_2'); grid on;

% Figure 2: Σφάλμα κατάστασης
figure;
plot(t,e_x(1,:),'k',t,e_x(2,:),'g'); grid on;
legend('e_1','e_2'); title('Σφάλμα e_x(t) = x(t) - \hat{x}(t)'); ylabel('e_x(t)'); xlabel('t [s]');

% Figure 3: Εκτιμήσεις A
figure;
for i = 1:n
    for j = 1:n
        subplot(n,n,(i-1)*n+j);
        plot(t, reshape(A_hat_all((i-1)*n+j,:),1,[]), 'b'); hold on;
        yline(A_true(i,j), 'r--');
        title(sprintf('a_{%d%d} estimate',i,j)); grid on;
    end
end

% Figure 4: Εκτιμήσεις B
figure;
for i = 1:n
    subplot(n,1,i);
    plot(t, B_hat_all(i,:), 'b'); hold on;
    yline(B_true(i), 'r--');
    title(sprintf('b_{%d} estimate',i)); grid on;
end
