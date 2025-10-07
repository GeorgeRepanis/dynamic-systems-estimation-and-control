clc; clear; close all;

% Χρονικές παράμετροι
Ts = 0.001;
Tfinal = 20;
t = 0:Ts:Tfinal;

% Πίνακες του συστήματος από εκφώνηση
A_true = [-2.15, 0.25; -0.75, -2];
B_true = [0; 1.5];

% Αρχικές συνθήκες
x = zeros(2, length(t));
x(:,1) = [0.5; -0.5];         % μη μηδενική αρχική κατάσταση
x_hat = zeros(2, length(t));
x_hat(:,1) = x(:,1);          % εκκίνηση εκτιμητή ίση με x(0)

% Εκτιμητής παραμέτρων
Theta_hat = zeros(2,3);
Theta_history = zeros(2,3,length(t));
gamma = 0.1;                  % πιο ασφαλής learning rate

% Είσοδος (ομαλή πολυαρμονική)
u = 0.6 * sin(2*pi*0.5*t) + 0.4 * sin(2*pi*1.1*t);

% Εκτιμητής - επανάληψη πραγματικού χρόνου
for k = 1:length(t)-1
    phi = [x(:,k); u(k)];
    dx = A_true * x(:,k) + B_true * u(k);    % πραγματική παράγωγος
    x(:,k+1) = x(:,k) + Ts * dx;

    dx_hat = (x(:,k+1) - x(:,k)) / Ts;
    x_hat(:,k+1) = Theta_hat * [x(:,k); u(k)];

    % Gradient ενημέρωση με φραγμό (clipping)
    Theta_hat = Theta_hat + gamma * (dx_hat - Theta_hat * phi) * phi';
    Theta_hat = max(min(Theta_hat, 50), -50);  % περιορισμός τιμών

    Theta_history(:,:,k) = Theta_hat;
end

% Τελικό σφάλμα κατάστασης
e = x - x_hat;

% === ΓΡΑΦΙΚΑ ===

% Είσοδος και καταστάσεις
figure;
subplot(3,1,1);
plot(t, u, 'm'); title('Είσοδος u(t) - πολυαρμονική'); ylabel('u(t)'); grid on;

subplot(3,1,2);
plot(t, x(1,:), 'b', t, x_hat(1,:), 'r--'); ylabel('x_1, \hat{x}_1'); legend('x_1', '\hat{x}_1'); grid on;

subplot(3,1,3);
plot(t, x(2,:), 'b', t, x_hat(2,:), 'r--'); xlabel('t [s]'); ylabel('x_2, \hat{x}_2'); legend('x_2', '\hat{x}_2'); grid on;

% Σφάλμα κατάστασης
figure;
plot(t, e(1,:), 'k', t, e(2,:), 'g'); xlabel('t [s]'); ylabel('e_x(t)'); legend('e_1', 'e_2'); title('Σφάλμα κατάστασης'); grid on;

% Εκτιμήσεις A
figure;
subplot(2,2,1); plot(t, squeeze(Theta_history(1,1,:)), 'b'); yline(-2.15, 'r--'); title('a_{11} estimate'); grid on;
subplot(2,2,2); plot(t, squeeze(Theta_history(1,2,:)), 'b'); yline(0.25, 'r--'); title('a_{12} estimate'); grid on;
subplot(2,2,3); plot(t, squeeze(Theta_history(2,1,:)), 'b'); yline(-0.75, 'r--'); title('a_{21} estimate'); grid on;
subplot(2,2,4); plot(t, squeeze(Theta_history(2,2,:)), 'b'); yline(-2, 'r--'); title('a_{22} estimate'); grid on;

% Εκτιμήσεις B
figure;
subplot(2,1,1); plot(t, squeeze(Theta_history(1,3,:)), 'b'); yline(0, 'r--'); title('b_1 estimate'); grid on;
subplot(2,1,2); plot(t, squeeze(Theta_history(2,3,:)), 'b'); yline(1.5, 'r--'); title('b_2 estimate'); grid on;
