% -------------------------------
% ΛΥΣΗ ΘΕΜΑΤΟΣ 2: Επιλογή Δομής και Αξιολόγηση Μοντέλου - MATLAB
% -------------------------------
% Περιλαμβάνει: Δημιουργία συνθετικών δεδομένων, ορισμό 3 μοντέλων,
% εκπαίδευση με Least Squares, εγκάρσια αξιολόγηση, και RLS σε ειδική βάση
% -------------------------------

% Ρυθμίσεις
T = 20; dt = 0.01; t = 0:dt:T; N = length(t);
theta1 = 1.5; theta2 = 1.0;
u = sin(t); x = zeros(1, N);

% Προσομοίωση συστήματος
for k = 1:N-1
    dx = -x(k)^3 + theta1 * tanh(x(k)) + theta2 / (1 + x(k)^2) + u(k);
    x(k+1) = x(k) + dx * dt;
end
dxdt = gradient(x, dt);

% Διαχωρισμός σε training/test
split_idx = floor(0.8 * N);
x_train = x(1:split_idx)'; u_train = u(1:split_idx)'; dxdt_train = dxdt(1:split_idx)';
x_test = x(split_idx+1:end)'; u_test = u(split_idx+1:end)'; dxdt_test = dxdt(split_idx+1:end)';

% -------------------------------
% Μοντέλο 1: Πολυωνυμικό (5ου βαθμού)
% -------------------------------
phi_poly_train = [x_train, x_train.^2, x_train.^3, x_train.^4, x_train.^5, u_train];
phi_poly_test = [x_test, x_test.^2, x_test.^3, x_test.^4, x_test.^5, u_test];
w_poly = phi_poly_train \ dxdt_train;
dxdt_poly = phi_poly_test * w_poly;
rmse_poly = sqrt(mean((dxdt_test - dxdt_poly).^2));

% -------------------------------
% Μοντέλο 2: RBF (7 βάσεις)
% -------------------------------
N_centers = 7; centers = linspace(min(x), max(x), N_centers); sigma = 0.5;
phi_rbf_train = exp(-(x_train - centers).^2 / (2 * sigma^2));
phi_rbf_test = exp(-(x_test - centers).^2 / (2 * sigma^2));
phi_rbf_train_full = [phi_rbf_train, u_train];
phi_rbf_test_full = [phi_rbf_test, u_test];
w_rbf = phi_rbf_train_full \ dxdt_train;
dxdt_rbf = phi_rbf_test_full * w_rbf;
rmse_rbf = sqrt(mean((dxdt_test - dxdt_rbf).^2));

% -------------------------------
% Μοντέλο 3: Ειδική βάση
% -------------------------------
phi_spec_train = [x_train, x_train.^2, x_train.^3, tanh(x_train), 1./(1+x_train.^2), u_train];
phi_spec_test = [x_test, x_test.^2, x_test.^3, tanh(x_test), 1./(1+x_test.^2), u_test];
w_spec = phi_spec_train \ dxdt_train;
dxdt_spec = phi_spec_test * w_spec;
rmse_spec = sqrt(mean((dxdt_test - dxdt_spec).^2));

% -------------------------------
% RLS στην ειδική βάση
% -------------------------------
w_rls = zeros(size(phi_spec_train,2),1); P = 1000*eye(size(phi_spec_train,2)); lambda = 0.99;
dxdt_rls_train = zeros(split_idx,1); w_hist = zeros(split_idx,length(w_rls));

for i = 1:split_idx
    phi_i = phi_spec_train(i,:)'; y_i = dxdt_train(i);
    K = (P*phi_i)/(lambda + phi_i'*P*phi_i);
    w_rls = w_rls + K*(y_i - phi_i'*w_rls);
    P = (P - K*phi_i'*P)/lambda;
    w_hist(i,:) = w_rls';
    dxdt_rls_train(i) = phi_i'*w_rls;
end

% Τελική πρόβλεψη με RLS
phi_spec_test = [x_test, x_test.^2, x_test.^3, tanh(x_test), 1./(1+x_test.^2), u_test];
dxdt_rls_test = phi_spec_test * w_rls;
rmse_rls = sqrt(mean((dxdt_test - dxdt_rls_test).^2));

% -------------------------------
% Αποτελέσματα
% -------------------------------
fprintf('\n--- RMSE Αποτελέσματα (Test Set) ---\n');
fprintf('Πολυωνυμικό:    %.5f\n', rmse_poly);
fprintf('RBF:            %.5f\n', rmse_rbf);
fprintf('Ειδική βάση:    %.5f\n', rmse_spec);
fprintf('RLS (Ειδική):   %.5f\n', rmse_rls);

% -------------------------------
% Γραφήματα
% -------------------------------
figure;
plot(t(split_idx+1:end), dxdt_test, 'k', 'LineWidth', 1.5); hold on;
plot(t(split_idx+1:end), dxdt_poly, 'g--');
plot(t(split_idx+1:end), dxdt_rbf, 'm--');
plot(t(split_idx+1:end), dxdt_spec, 'b--');
plot(t(split_idx+1:end), dxdt_rls_test, 'r-.');
legend('True', 'Poly', 'RBF', 'Special', 'RLS Special');
title('Σύγκριση Προβλέψεων στο Test Set'); xlabel('Time [s]'); ylabel('dx/dt'); grid on;

figure;
plot(t(1:split_idx), w_hist);
title('Εξέλιξη Παραμέτρων RLS - Ειδική Βάση'); xlabel('Time [s]'); ylabel('w_i(t)'); grid on;
legend({'w_1: x', 'w_2: x^2', 'w_3: x^3', 'w_4: tanh(x)', 'w_5: 1/(1+x^2)', 'w_6: u'}, 'Location', 'best');