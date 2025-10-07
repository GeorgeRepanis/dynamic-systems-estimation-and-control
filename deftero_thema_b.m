% -------------------------------
% Τελική αξιολόγηση RLS με νέο test set & υπολογισμός ενεργειακού σφάλματος
% -------------------------------

% Ρυθμίσεις
T = 20; dt = 0.01; t = 0:dt:T; N = length(t);
theta1 = 1.5; theta2 = 1.0;
u = sign(sin(0.5 * pi * t));  % Τετραγωνικό κύμα 
x = zeros(1, N);

% Προσομοίωση συστήματος με νέο u(t)
for k = 1:N-1
    dx = -x(k)^3 + theta1 * tanh(x(k)) + theta2 / (1 + x(k)^2) + u(k);
    x(k+1) = x(k) + dx * dt;
end
dxdt = gradient(x, dt);

% Διαχωρισμός σε training/test
split_idx = floor(0.8 * N);
x_train = x(1:split_idx)'; u_train = u(1:split_idx)'; dxdt_train = dxdt(1:split_idx)';
x_test = x(split_idx+1:end)'; u_test = u(split_idx+1:end)'; dxdt_test = dxdt(split_idx+1:end)';

% Ειδική βάση
phi_spec_train = [x_train, x_train.^2, x_train.^3, tanh(x_train), 1./(1+x_train.^2), u_train];
phi_spec_test = [x_test, x_test.^2, x_test.^3, tanh(x_test), 1./(1+x_test.^2), u_test];

% RLS
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

% Πρόβλεψη στο νέο test set
phi_spec_test = [x_test, x_test.^2, x_test.^3, tanh(x_test), 1./(1+x_test.^2), u_test];
dxdt_rls_test = phi_spec_test * w_rls;

% Υπολογισμός ενεργειακού σφάλματος
e_test = dxdt_test - dxdt_rls_test;
e = sum(e_test.^2) * dt;  % Προσέγγιση ολοκληρώματος

% Αποτελέσματα
fprintf('\n>>> Τελική αξιολόγηση RLS με νέο test set <<<\n');
fprintf('Συνολικό σφάλμα e: %.5f\n', e);

% Γραφήματα
figure;
plot(t(split_idx+1:end), dxdt_test, 'k', 'LineWidth', 1.5); hold on;
plot(t(split_idx+1:end), dxdt_rls_test, 'r--');
title('Πρόβλεψη με RLS σε νέο Test Set'); xlabel('Time [s]'); ylabel('dx/dt'); grid on;
legend('True', 'RLS Prediction');

figure;
plot(t(1:split_idx), w_hist);
title('Εξέλιξη Παραμέτρων RLS (Νέο Test Set)'); xlabel('Time [s]'); ylabel('w_i(t)'); grid on;
legend({'w_1: x', 'w_2: x^2', 'w_3: x^3', 'w_4: tanh(x)', 'w_5: 1/(1+x^2)', 'w_6: u'}, 'Location', 'best');
