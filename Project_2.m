% Sohini Gupta 20EE38032
% Computational Neuroscience EC60007
% Project 2

clear all;
close all;
warning off;
%% Approximate run time : 2 minutes
%% model initialization
global t_start;
global t_end;
global GCa;
global GK;
global GL;
global VCa;
global VK;
global VL;
global phi;
global V1;
global V2;
global V3;
global V4;
global V5;
global V6;
global C;
global Iext;
%% setting parameter values
t_start = cputime;
C = 20 ;    %microfarad/cm^2 
GCa=4.4;    %millisiemens/cm^2 
GK=8;       %millisiemens/cm^2
GL=2;       %millisiemens/cm^2
VCa=120;    %millivolts
VK=-84;     %millivolts
VL=-60;     %millivolts
phi = 0.02; %per millisecond
V1=-1.2;    %millivolts
V2= 18 ;    %millivolts
V3= 2 ;     %millivolts
V4= 30;     %millivolts
V5= 2;      %not found
V6= 20;     %not found
Iext=0;     %microamps/cm^2
%% Generating phase plane, null clines and intersection
fprintf('\n ------------------------- Part 1 ------------------------------\n ');
fprintf('\n ---Consistent units have been taken and consistency verified---\n ');
fprintf('\n ------------------------- Part 2 ------------------------------\n ');
fprintf('\n ------Phase plane, null clines and equilibrium point ---------------\n ');
% generate nullclines
figure;
hold on;

V_null = @(V) (Iext - GCa*(0.5*(1+tanh((V-V1)/V2)))*(V-VCa) - GL*(V-VL))/(GK*(V-VK));
w_null = @(V) 0.5*(1+tanh((V-V3)/V4)); % same as w_inf(V)
fplot(@(V)100*V_null(V), [-80 130],"LineWidth",1.5);  % w plotted as a percentage
fplot(@(V)100*w_null(V), [-80 130],"LineWidth",1.5);  % instead of probability
legend("V null cline","w null cline");
xlabel('V(in mV)');
ylabel('w in %');
title('Moris Lecar Equations Null clines');
grid on;

% find equilibrium points
syms V w;
m_inf = 0.5*(1+tanh((V-V1)/V2));
w_inf = 0.5*(1+tanh((V-V3)/V4));
V_null_eqn = (1/C)*(Iext - GCa*m_inf*(V-VCa) - GK*w*(V-VK) - GL*(V-VL)) == 0;
w_null_eqn = (w - w_inf) == 0;
eq_point = solve([V_null_eqn, w_null_eqn], [V, w]);

V_eq = double(eq_point.V);
w_eq = double(eq_point.w);

plot(V_eq,100*w_eq, '*', 'linewidth', 2);
text(V_eq,100*w_eq, ['(' num2str(round(V_eq,3)) ',' num2str(round(w_eq,3)) ')']);
grid on;

fprintf('\n The equilibrium point is located at (%d,%d) \n', V_eq, w_eq);
% Simulate Trajectories
x = linspace(-80,120,200);
y = linspace(0,1,200);

[V_quiver,w_quiver] = meshgrid(x,y);
m_inf = 0.5*(1+tanh((V_quiver-V1)/V2));
w_inf = 0.5*(1+tanh((V_quiver-V3)/V4));
tau_w_inv = cosh((V_quiver-V3)/(2*V4));
dV = (1/C)*(Iext - GCa*m_inf.*(V_quiver-VCa) - GK*w_quiver.*(V_quiver-VK) - GL*(V_quiver-VL));
dw = phi*(w_inf-w_quiver).*tau_w_inv;
quiver(x,y*100,dV,dw*100);
legend('V null cline', 'w null cline','Equilibrium Point','Trajectories');


%% Jacobian at a point near equilibrium
syms V w;
m_inf = 0.5*(1+tanh((V-V1)/V2));
w_inf = 0.5*(1+tanh((V-V3)/V4));
tau_w_inv = cosh((V-V3)/(2*V4));
dV = (1/C)*(Iext - GCa*m_inf.*(V-VCa) - GK*w.*(V-VK) - GL*(V-VL));
dw = phi*(w_inf-w)*tau_w_inv;

J = jacobian([dV, dw],[V,w]);
V = V_eq;
w = w_eq;
J_mat = zeros(2,2);
J_mat(1,1) = subs(J(1,1));
J_mat(1,2) = subs(J(1,2));
J_mat(2,1) = subs(J(2,1));
J_mat(2,2) = subs(J(2,2));

eigen_values = eig(J_mat);
fprintf('\n------------------------- Part 3 ------------------------------\n ');
fprintf('The eigen values are %d and %d \n', eigen_values(1), eigen_values(2));
if eigen_values(1)<0 && eigen_values(2)<0
fprintf('\n--------------------stable equilibrium-------------------------\n ');
end
fprintf('\n-------------------------------Part 4 -----------------------------------\n ');
fprintf('\n--Voltage (kV) would require AbsTol = 10^-12. Will slow down the solver--\n ');
%% Trajectories added to phase plane plot
fprintf('\n------------------------- Part 5 ------------------------------\n ');
options = odeset('RelTol',1e-3,'AbsTol',1e-6, 'refine',5, 'MaxStep', 1);

Iext = 0;
time = [0, 300];
del_V = 60; %millivolts
initial_conditions = [V_eq + del_V,w_eq];
phi = 0.01;
[t1, S1] = ode15s(@(t,S)mle_gradient(t,S),time, initial_conditions, options);
phi = 0.02;
[t2, S2] = ode15s(@(t,S)mle_gradient(t,S),time, initial_conditions, options);
phi = 0.04;
[t3, S3] = ode15s(@(t,S)mle_gradient(t,S),time, initial_conditions, options);
phi = 0.02;  % changing back to default value after computing trajectories

figure;
hold on;
plot(t1,S1(:,1));
plot(t2, S2(:,1));
plot(t3, S3(:,1));
xlabel('Time(in ms)');
ylabel('Voltage(in mV)');
title('Action potentials with different \phi');
legend('\phi = 0.01','\phi = 0.02','\phi = 0.04');
grid on;

figure;
hold on;
V_null = @(V) (Iext - GCa*(0.5*(1+tanh((V-V1)/V2)))*(V-VCa) - GL*(V-VL))/(GK*(V-VK));
w_null = @(V) 0.5*(1+tanh((V-V3)/V4));
fplot(@(V) V_null(V), [-80 100],'k');
fplot(@(V) w_null(V), [-80 100],'k');

plot(S1(:,1),S1(:,2));
plot(S2(:,1),S2(:,2));
plot(S3(:,1),S3(:,2));
xlabel('V(in mV)');
ylabel('w');
ylim([0,1]);
title('Phase Plane Plot(MLE)');
legend('V null cline','w null cline','\phi = 0.01','\phi = 0.02','\phi = 0.04');
grid on;

%% Threshold behaviour
fprintf('\n------------------------- Part 6 ------------------------------\n ');
Iext = 0;
time = [0, 300];
phi=0.02;
num_initial = 450; % how many initial membrane potentials to simulate
V_init=linspace(-20,-10,num_initial); % approximate range obtained visually
V_max = zeros(1,num_initial);
figure;
hold on;
title("Soft threshold behaviour");
xlabel("Time in ms");
ylabel("Membrane potential");
grid on;
flag=1;
for i = 1:num_initial
    [t,S] = ode15s(@(t,S)mle_gradient(t,S),time,[V_init(i),w_eq], options);
    if mod(i,30)==0
        plot(t,S(:,1));
    end
    V_max(i) = max(S(:,1));
    if V_max(i) >= 0 && flag==1  % records the first value of V_init to result in an action potential
        fprintf("\nThreshold is %f mV\n",V_init(i));
        threshold = V_init(i);
        flag=0;
    end
end

figure;
hold on;
plot(V_init,V_max);
grid on;
xlabel('Initial Voltage(in mV)');
ylabel('Maximum Voltage(in mV)');
title('Threshold behavior');

figure;
hold on;
V_null = @(V) (Iext - GCa*(0.5*(1+tanh((V-V1)/V2)))*(V-VCa) - GL*(V-VL))/(GK*(V-VK));
w_null = @(V) (0.5*(1+tanh((V-V3)/V4)));
fplot(@(V) V_null(V), [-80 100],'k');
fplot(@(V) w_null(V), [-80 100],'k');

V_plot = linspace(threshold-0.1,threshold + 0.1,5);
for i = 1:5
    [t,S] = ode15s(@(t,S)mle_gradient(t,S),time, [V_plot(i),w_eq], options);
    plot(S(:,1),S(:,2));
end
legend(num2str(V_plot(1)),num2str(V_plot(2)),num2str(V_plot(3)),num2str(V_plot(4)),num2str(V_plot(5)));
xlabel('V(in mV)');
ylabel('w');
ylim([0,1]);
title('Phase Plane Plot(MLE) for different initial voltages around threshold');
grid on;
%% Trajectories during dc current injection
fprintf('\n------------------------- Part 7 ------------------------------\n ');
Iext = 86;
figure;
hold on;
V_null = @(V) (Iext - GCa*(0.5*(1+tanh((V-V1)/V2)))*(V-VCa) - GL*(V-VL))/(GK*(V-VK));
w_null = @(V) 0.5*(1+tanh((V-V3)/V4)); % same as w_inf(V)
fplot(@(V)V_null(V), [-80 120],"LineWidth",1.5);  % w plotted as a percentage
fplot(@(V)w_null(V), [-80 120],"LineWidth",1.5);  % instead of probability
legend("V null cline","w null cline");
xlabel('V(in mV)');
ylabel('w');
ylim([0 1]);
title('Moris Lecar Equations Null clines');
grid on;

% computing equilibrium for dc current injection
syms V w;
m_inf = 0.5*(1+tanh((V-V1)/V2));
w_inf = 0.5*(1+tanh((V-V3)/V4));
V_null_eqn = (1/C)*(Iext - GCa*m_inf*(V-VCa) - GK*w*(V-VK) - GL*(V-VL)) == 0;
w_null_eqn = (w - w_inf) == 0;
eq_point = solve([V_null_eqn, w_null_eqn], [V, w]);
V_eq_dc = double(eq_point.V);
w_eq_dc = double(eq_point.w);

plot(V_eq_dc, w_eq_dc, '*', 'linewidth', 1.5);
text(V_eq_dc, w_eq_dc, ['(' num2str(round(V_eq_dc,3)) ',' num2str(round(w_eq_dc,3)) ')']);
fprintf('The equilibrium point is located at (%d,%d) \n', V_eq_dc, w_eq_dc);
grid on;

syms V w;
m_inf = 0.5*(1+tanh((V-V1)/V2));
w_inf = 0.5*(1+tanh((V-V3)/V4));
tau_w_inv = cosh((V-V3)/(2*V4));
dV = (1/C)*(Iext - GCa*m_inf.*(V-VCa) - GK*w.*(V-VK) - GL*(V-VL));
dw = phi*(w_inf-w)*tau_w_inv;

J = jacobian([dV, dw],[V,w]);
V = V_eq_dc;
w = w_eq_dc;
J_mat = zeros(2,2);
J_mat(1,1) = subs(J(1,1));
J_mat(1,2) = subs(J(1,2));
J_mat(2,1) = subs(J(2,1));
J_mat(2,2) = subs(J(2,2));

eigen_values = eig(J_mat);
fprintf('The real parts of eigen values are %d and %d \n', eigen_values(1), eigen_values(2));
if eigen_values(1)<0 && eigen_values(2)<0
fprintf('\n--------------------stable equilibrium-------------------------\n ');
end

time = [0 900];
init_1 = [V_eq,w_eq];
[t,S] = ode15s(@(t,S)mle_gradient(t,S),time,init_1, options);
init_2 = [V_eq_dc,w_eq_dc];
[t1,S2] = ode15s(@(t,S)mle_gradient(t,S),time,init_2, options);
init_3 = [-27.9,0.17];
[t3,S3] = ode15s(@(t,S)mle_gradient(t,S),time,init_3, options);
plot(S(:,1),S(:,2));
plot(S2(:,1),S2(:,2));
plot(S3(:,1),S3(:,2));
legend('V - nullcline','w - nullcline','Equilibrium point','Eq pt of Ixt = 0','Eq pt of Ixt = 86','Random intial point');
figure;
hold on;
plot(t,S(:,1));
plot(t1, S2(:,1));
plot(t3, S3(:,1));
xlabel('Time(in ms)');
ylabel('Voltage(in mV)');
title('Trajectories with different initial conditions');
legend('Eq pt of Ixt = 0','Eq pt of Ixt = 86','Random initial point');
grid on;
%% Unstable Periodic Orbit
Iext = 86;
figure;
hold on;
V_null = @(V) (Iext - GCa*(0.5*(1+tanh((V-V1)/V2)))*(V-VCa) - GL*(V-VL))/(GK*(V-VK));
w_null = @(V) 0.5*(1+tanh((V-V3)/V4)); % same as w_inf(V)
fplot(@(V)V_null(V), [-80 120],"LineWidth",1.5);  % w plotted as a percentage
fplot(@(V)w_null(V), [-80 120],"LineWidth",1.5);  % instead of probability
legend("V null cline","w null cline");
xlabel('V(in mV)');
ylabel('w');
ylim([0 1]);
title('Moris Lecar Equations Null clines');
grid on;

plot(V_eq_dc, w_eq_dc, 'o');
text(V_eq_dc, w_eq_dc, ['(' num2str(round(V_eq_dc,3)) ',' num2str(round(w_eq_dc,3)) ')']);
grid on;
time = [0 300];
rev_time = [0 -1000];
[~,S] = ode15s(@(t,S)mle_gradient(t,S),time,[V_eq,w_eq],options);
[~,S2] = ode15s(@(t,S)mle_gradient(t,S),time,[-27.9, 0.17],options);
[~,S3] = ode15s(@(t,S)mle_gradient(t,S),time,[V_eq_dc, w_eq_dc],options);
[~,S4] = ode15s(@(t,S)mle_gradient(t,S),rev_time,[-27.9,0.17],options);

plot(S(:,1),S(:,2));
plot(S2(:,1),S2(:,2));
plot(S3(:,1),S3(:,2));
plot(S4(:,1),S4(:,2));
legend('V - nullcline','w - nullcline','Equilibrium point','Eq pt of Ixt = 0','Eq pt of Ixt = 86','Random intial point','reverse trajectory for random initial point ');

w_init = 0.13;
num_initial = 500;
V_init = linspace(-21.34,-21.3,num_initial);
V_max = zeros(1,num_initial);
flag=1;
for i = 1:num_initial
    [t,S] = ode15s(@(t,S)mle_gradient(t,S),time,[V_init(i),w_init], options);
    V_max(i) = max(S(:,1));
end
figure;
plot(V_init,V_max);
title("True threshold behaviour");
xlabel("Initial voltage in mV");
ylabel("Peak voltage in mV");
grid on;
figure;
hold on;
time = [0 600];
[t,S] = ode15s(@(t,S)mle_gradient(t,S),time,[-21.32,w_init],options);
[t1,S2] = ode15s(@(t,S)mle_gradient(t,S),time,[-21.31,w_init],options);
plot(t,S(:,1));
plot(t1,S2(:,1));
title("Membrane potential variation due to minute difference in initial condition");
xlabel("Time in sec");
ylabel("Voltage in mV");
legend("V initial = -21.32 mV","V initial = -21.31 mV");
grid on;
%% Response for Iext = 80, 86 and 90 uA/cm^2
I_vec = [80, 86, 90];
for i = 1:length(I_vec)
    Iext = I_vec(i);
    fprintf('\n------------------------- Part 9 -> Iext = %d uA/cm^2------------------------- \n', Iext);
    % Finding the equilibrium point
    syms V w;
    m_inf = 0.5*(1+tanh((V-V1)/V2));
    w_inf = 0.5*(1+tanh((V-V3)/V4));
    V_null = (1/C)*(Iext - GCa*m_inf*(V-VCa) - GK*w*(V-VK) - GL*(V-VL)) == 0;
    w_null = (w_inf - w) == 0;
    eq_pt = solve([V_null, w_null], [V, w]);

    V_eq = double(eq_pt.V);
    w_eq = double(eq_pt.w);
    fprintf('The equilibrium point is located at (%d,%d)  \n', V_eq, w_eq);
    % Stability Analysis
    syms V w;
    m_inf = 0.5*(1+tanh((V-V1)/V2));
    w_inf = 0.5*(1+tanh((V-V3)/V4));
    tau_w_inv = cosh((V-V3)/(2*V4));
    dV = (1/C)*(Iext - GCa*m_inf.*(V-VCa) - GK*w.*(V-VK) - GL*(V-VL));
    dw = phi*(w_inf-w)*tau_w_inv;

    J = jacobian([dV, dw],[V,w]);
    V = V_eq;
    w = w_eq;
    J_mat = zeros(2,2);
    J_mat(1,1) = subs(J(1,1));
    J_mat(1,2) = subs(J(1,2));
    J_mat(2,1) = subs(J(2,1));
    J_mat(2,2) = subs(J(2,2));
    
    eigenValues = eig(J_mat);
    fprintf('The eigen values are  %f%+fi , %f%+fi \n', real(eigenValues(1)), imag(eigenValues(1)),real(eigenValues(2)), imag(eigenValues(2)));
end
% Trajectories for different Iext starting near equilibrium
figure;
hold on;
title("Voltage waveform for different Iext");
xlabel("Time in ms");
ylabel("Membrane Potential in mV");
grid on;
i = 0;
I_vec = 80:100;%uA/cm^2
num_I = length(I_vec);
frequency = zeros(num_I, 1);
current = zeros(num_I, 1);
options = odeset('RelTol',1e-3,'AbsTol',1e-6, 'refine',5, 'MaxStep', 1);
for Iext = I_vec
    syms V w;
    m_inf = 0.5*(1+tanh((V-V1)/V2));
    w_inf = 0.5*(1+tanh((V-V3)/V4));
    V_null = (1/C)*(Iext - GCa*m_inf*(V-VCa) - GK*w*(V-VK) - GL*(V-VL)) == 0;
    w_null = (w_inf - w) == 0;
    eq_pt = solve([V_null, w_null], [V, w]);
    V_eq = double(eq_pt.V);
    w_eq = double(eq_pt.w);
    
    initial_conditions = [V_eq + 0.1, w_eq + 0.001];
    time = [0 2000];
    [t,S]=ode15s(@(t,S)mle_gradient(t,S),time,initial_conditions,options);
    if Iext == 80 || Iext == 86 || Iext == 90
        plot(t,S(:,1));
    end
    i = i+1;
    frequency(i) = spike_frequency(t, S);
    current(i) = Iext;
end
legend({'V_m for 80uA/cm^2','V_m for 86uA/cm^2','V_m for 90uA/cm^2'},"Location","northwest");
figure;
grid on;
hold on;
plot(current, frequency,"b");
ylabel('Firing Rate in Hz');
xlabel('Iext (in uA/cm^2)');
title('Firing Rate vs Current Injection');
hold off;

%% New parameters (Saddle point)
% Setting new parameter values
GCa = 4;
GK = 8;
GL = 2;
VCa = 120;
VK = -84;
VL = -60;
phi = 0.0667;
V1 = -1.2;
V2 = 18;
V3 = 12;
V4 = 17.4;
C = 20;
Iext = 30;
%% Phase Plane with V and w null clines
fprintf('\n------------------------- Part 10 ------------------------------ \n');
figure;
hold on;
V_null = @(V) (Iext - GCa*(0.5*(1+tanh((V-V1)/V2)))*(V-VCa) - GL*(V-VL))/(GK*(V-VK));
w_null = @(V) (0.5*(1+tanh((V-V3)/V4)));
fplot(@(V) V_null(V), [-80 100],"LineWidth",1);
fplot(@(V) w_null(V), [-80 100],"LineWidth",1);
ylim([-0.1 1]);
xlabel('V(in mV)');
ylabel('w');
title('Moris Lecar Equations Phase Plane');

% Finding and plotting the equilibrium point for this system using MATLAB
syms V w;
m_inf = 0.5*(1+tanh((V-V1)/V2));
w_inf = 0.5*(1+tanh((V-V3)/V4));
V_null = (1/C)*(Iext - GCa*m_inf*(V-VCa) - GK*w*(V-VK) - GL*(V-VL)) == 0;
w_null = (w_inf - w) == 0;

% initial guess obtained by visual inspection
soln1 = vpasolve([V_null, w_null], [V, w], [-40, 0]);
soln2 = vpasolve([V_null, w_null], [V, w], [-20, 0]);
soln3 = vpasolve([V_null, w_null], [V, w], [4, 0.3]);

V_eq1 = double(soln1.V);
w_eq1 = double(soln1.w);
plot(V_eq1, w_eq1, 'o',"LineWidth",2);
text(V_eq1, w_eq1, ['(' num2str(round(V_eq1,3)) ',' num2str(round(w_eq1,3)) ')']);
grid on;

V_eq2 = double(soln2.V);
w_eq2 = double(soln2.w);
plot(V_eq2, w_eq2, 'o',"LineWidth",2);
text(V_eq2, w_eq2, ['(' num2str(round(V_eq2,3)) ',' num2str(round(w_eq2,3)) ')']);
grid on;

V_eq3 = double(soln3.V);
w_eq3 = double(soln3.w);
plot(V_eq3, w_eq3, 'o',"LineWidth",2);
text(V_eq3, w_eq3, ['(' num2str(round(V_eq3,3)) ',' num2str(round(w_eq3,3)) ')']);
grid on;

fprintf('Equilibrium points are : \n 1. (%f, %f) \n 2. (%f, %f) \n 3. (%f, %f)\n', V_eq1, w_eq1, V_eq2, w_eq2, V_eq3, w_eq3);

% Stability ananlysis of equilibrium point:
syms V w;
m_inf = 0.5*(1+tanh((V-V1)/V2));
w_inf = 0.5*(1+tanh((V-V3)/V4));
tau_w_inv = cosh((V-V3)/(2*V4));
V_eq = [V_eq1, V_eq2, V_eq3];
w_eq = [w_eq1, w_eq2, w_eq3];
dV = (1/C)*(GCa*m_inf*(VCa-V) + GK*w*(VK-V) + GL*(VL-V) + Iext);
dw = phi*(w_inf - w)*tau_w_inv;
J = jacobian([dV, dw],[V,w]);
eVecSaddle = 0;
for i = 1:3
    V = V_eq(i);
    w = w_eq(i);
    J_mat = zeros(2,2);
    J_mat(1,1) = subs(J(1,1));
    J_mat(1,2) = subs(J(1,2));
    J_mat(2,1) = subs(J(2,1));
    J_mat(2,2) = subs(J(2,2));
    eigenValues = eig(J_mat);
    if i == 2
        [eVecSaddle, eValSaddle] = eig(J_mat);
    end
    fprintf('Equilibrium point %d : The eigen values are  %f%+fi , %f%+fi \n', i, real(eigenValues(1)), imag(eigenValues(1)),real(eigenValues(2)), imag(eigenValues(2)));
end

% Plot the manifolds of the saddle point:
vu = eVecSaddle(:,1)';  % eigen vector corresponding to positive eigen value
vs = eVecSaddle(:,2)';  % eigen vector corresponding to negative eigen value

% Evaluating manifolds
p = 1e-1;  % perturbation
saddle_eq = [V_eq2,w_eq2];
stable_up =     saddle_eq +  p*vs;
stable_down =   saddle_eq -  p*vs;
unstable_up =   saddle_eq +  p*vu;
unstable_down = saddle_eq -  p*vu;
options = odeset('RelTol',1e-3,'AbsTol',1e-6, 'refine',5, 'MaxStep', 1);


[~,S]=ode15s(@(t,S)mle_gradient(t,S), [0 -500], stable_up, options);
plot(S(:,1), S(:,2), 'g',"LineWidth",1);
[~,S]=ode15s(@(t,S)mle_gradient(t,S), [0 500], unstable_up, options);
plot(S(:,1), S(:,2), 'm',"LineWidth",1);
[~,S]=ode15s(@(t,S)mle_gradient(t,S), [0 -200], stable_down, options);
plot(S(:,1), S(:,2), 'g',"LineWidth",1);
[~,S]=ode15s(@(t,S)mle_gradient(t,S), [0 200], unstable_down, options);
plot(S(:,1), S(:,2), 'm',"LineWidth",1);
ylim([-0.2 1]);
legend('V nullcline','w nullcline','Eq : Stable', 'Eq: Saddle', 'Eq: Unstable','Stable manifolds', 'Unstable manifolds');
%% Firing rate vs current injection
fprintf('\n------------------------- Part 11 ------------------------------ \n');
% For Iext between 30 and 50 finding the nature of stability  of each equilibrium point
currents = [30, 35, 39, 39.1, 39.2, 39.3, 39.4, 39.5, 39.6, 39.7, 39.8, 39.9, 40, 41, 42, 45];
N = size(currents,2);
time = [0 2000];
frequency = zeros(N, 1);
for i = 1:N
    Iext = currents(i);
    figure;
    subplot(2,1,1);
    hold on;
    % Plotting the Phase Plane with V and w null clines
    V_null = @(V) (Iext - GCa*(0.5*(1+tanh((V-V1)/V2)))*(V-VCa) - GL*(V-VL))/(GK*(V-VK));
    w_null = @(V) (0.5*(1+tanh((V-V3)/V4)));
    fplot(@(V) V_null(V), [-80 100],"LineWidth",1);
    fplot(@(V) w_null(V), [-80 100],"LineWidth",1);
    xlabel('V(in mV)');
    ylabel('w');
    title(strcat('Phase Plane Plot(MLE) with Iext = ', num2str(Iext)));
    hold on;
    % Finding and plotting the equilibrium point for this system using MATLAB
    syms V w;
    V_null = (1/C)*(Iext - GCa*(0.5*(1+tanh((V-V1)/V2)))*(V-VCa) - GK*w*(V-VK) - GL*(V-VL)) == 0;
    w_null = (0.5*(1+tanh((V-V3)/V4)) - w) == 0;
    soln1 = vpasolve([V_null, w_null], [V, w], [-40, 0]);
    soln2 = vpasolve([V_null, w_null], [V, w], [-20, 0]);
    soln3 = vpasolve([V_null, w_null], [V, w], [0, 0.3]);

    V_eq1 = double(soln1.V);
    w_eq1 = double(soln1.w);
    V_eq2 = double(soln2.V);
    w_eq2 = double(soln2.w);
    V_eq3 = double(soln3.V);
    w_eq3 = double(soln3.w);
    
    if isequal([V_eq1, w_eq1], [V_eq2, w_eq2]) 
        fprintf("For Iext = %f, there is only one distinct equilibrium point\n", Iext);
        [t,S]=ode15s(@(t,S)mle_gradient(t,S),time, [V_eq3 + 0.1, w_eq3+0.01]);
        plot(V_eq3, w_eq3,".k","LineWidth",8);
        text(V_eq3, w_eq3, ['(' num2str(round(V_eq3,3)) ',' num2str(round(w_eq3,3)) ')']);
        grid on;
        plot(S(:,1), S(:,2));
        ylim([-0.2 1]);
        frequency(i) = spike_frequency(t, S);
        legend("V null cline","w null cline", "Eq: Unstable","Trajectory ending on limit cycle");
        subplot(2,1,2);
        plot(t,S(:,1));
        title("Membrane Potential vs time");
        xlabel("Time in seconds");
        ylabel("V membrane in mV");
        grid on;
    else
        fprintf("For Iext = %f, there are three distinct equilibrium points\n", Iext);
        [t,S]=ode15s(@(t,S)mle_gradient(t,S),time, [V_eq2 - 0.1, w_eq2-0.01]);
        frequency(i) = spike_frequency(t,S);
        plot(V_eq1, w_eq1,".r","LineWidth",8);
        text(V_eq1-20, w_eq1-0.05, ['(' num2str(round(V_eq1,3)) ',' num2str(round(w_eq1,3)) ')']);
        grid on;
        plot(V_eq2, w_eq2,".g","LineWidth",8);
        text(V_eq2+20, w_eq2-0.05, ['(' num2str(round(V_eq2,3)) ',' num2str(round(w_eq2,3)) ')']);
        grid on;
        plot(V_eq3, w_eq3,".b","LineWidth",8);
        text(V_eq3, w_eq3, ['(' num2str(round(V_eq3,3)) ',' num2str(round(w_eq3,3)) ')']);
        grid on;
        plot(S(:,1), S(:,2));
        ylim([-0.2 1]);
        legend("V null cline","w null cline", "Eq: Stable","Eq: Saddle","Eq: Unstable","Trajectory");   
        subplot(2,1,2);
        plot(t,S(:,1));
        title("Membrane Potential vs time");
        xlabel("Time in seconds");
        ylabel("V membrane in mV");
        grid on;
    end
end
hold off;
figure;
hold on;
ylabel('Firing Rate in Hz');
xlabel('Iext in uA/cm^2');
title('Firing Rate vs Current injection');
plot(currents, frequency);
hold off;
%% Hodgin Huxley Equations parameters initialization    
fprintf('\n------------------------- Part 12 ------------------------------ \n');
fprintf('\n------------Parameters for Hodgin Huxley Equations---------------\n');
warning off;
global C;
global Iext;
global GK;
global GNa;
global GL;
global VK;
global VNa;
global VL;
global epsilon;
global h_inf_Vr;
GNa = 120;
GK = 36;
VNa = 55;
VK = -72;
GL = 0.3;
Iext = 0;
C = 1;
epsilon = 1e-9;
options = odeset('RelTol',1e-10,'AbsTol',1e-10, 'refine',5, 'MaxStep', 1);
%% Finding E_L for Vr = -60mV
fprintf('\n------------------------- Part 13 ------------------------------ \n');
Vr = -60;
alphan = -0.01 * (Vr + epsilon + 50)/(exp(-(Vr + epsilon + 50)/10)-1);
alpham = -0.1 * (Vr + epsilon + 35)/(exp(-(Vr + epsilon + 35)/10)-1);
alphah = 0.07 * exp(-(Vr + 60)/20);
betan = 0.125 * exp(-(Vr + 60)/80);
betam = 4 * exp(-(Vr + 60)/18);
betah = 1/(exp(-(Vr + 30)/10) + 1);    
m_inf = alpham/(alpham + betam);
n_inf = alphan/(alphan + betan);
h_inf = alphah/(alphah + betah);
h_inf_Vr = h_inf;
E_leak = Vr - (1/GL)*(Iext - GK * (n_inf^4) * (Vr - VK) - GNa * (m_inf^3) * h_inf * (Vr - VNa));
VL = E_leak;
fprintf('E_L = %0.2f mV\n', E_leak);

 %% Iext = 10uA
Iext = 10;
time = [0 100];    
initial_condition = [-60, n_inf, m_inf, h_inf]; 
[t, S] = ode15s(@(t,S)hhe_gradient(t,S), time, initial_condition, options);
figure;
plot(t, S(:,1));
xlabel('Time(ms)');
ylabel('Voltage (in mV)');
title('Response for 10uA/cm^2 current injection');
grid on;
%% stability for Iext = 0 uA/cm^2
fprintf("\nIext = 0 uA/cm^2\n");
stability_analysis(0);
%% current injection threshold
fprintf('\n------------------------- Part 14 ------------------------------ \n');
Iext = 0;
N = 100;
current = linspace(0,20,N);
V_max = zeros(N,1);
time = [0 1000];
for i = 1:N 
    initial_condition = [Vr + (current(i)/C), n_inf, m_inf, h_inf];
    [t, S] = ode15s(@(t,S)hhe_gradient(t,S), time, initial_condition, options);
    V_max(i) = max(S(:,1));
end
V_max_shifted = [0;V_max(1:end-1)];
dV_max = V_max-V_max_shifted;
[~,idx_threshold] = max(dV_max);
current_threshold = (current(idx_threshold)+current(idx_threshold-1))/2;
fprintf("Threshold: %f uA/cm^2\n",current_threshold);
figure;
plot(current, V_max);    
xlabel('impulse current(uA/cm^2)');
ylabel('Peak Voltage (in mV)');
title('Current pulse Threshold');
grid on;
%% Stability for Iext = 8-12 uA/cm^2
fprintf('\n------------------------- Part 15 ------------------------------ \n');  
for i=8:12
    fprintf("\nIext = %d", i);
    stability_analysis(i);
end
%% Behaviour of complete Hodgin Huxley model with varying current injection
fprintf('\n------------------------- Part 16 ------------------------------ \n'); 
Iext = 10;
time = [0 100];    
initial_condition = [Vr, n_inf, m_inf, h_inf]; 
[t, S] = ode15s(@(t,S)hhe_gradient(t,S), time, initial_condition, options);
figure;
plot(t, S(:,1));
xlabel('Time(ms)');
ylabel('Voltage (in mV)');
title('Response of complete HH model to 10uA/cm^2 current injection');
grid on;

% Response for current impulse. (includes values before/after threshold)
Iext = 0;
currents = linspace(0,15,5);
time = [0 25];
figure;
hold on;
xlabel('Time(ms)');
ylabel('Voltage (in mV)');
title('Simulating complete HH model for current pulse');
for i = 1:5
    initial_condition = [-60+currents(i)/C, n_inf,m_inf,h_inf];
    [t, S] = ode15s(@(t,S)hhe_gradient(t,S),time, initial_condition, options);
    plot(t, S(:,1));
    hold on;
end
legend(strcat('impulse = ', num2str(currents(1))), strcat('impulse = ', num2str(currents(2))),strcat('impulse = ', num2str(currents(3))), strcat('impulse = ', num2str(currents(4))),strcat('impulse = ', num2str(currents(5))));
hold off;
%% Behaviour of Reduced V-n Model
fprintf('\n---------------------Part 16 continued-----------------------\n');
Iext = 10;
time = [0 100];    
initial_condition = [-60, n_inf]; 
[t, S] = ode15s(@(t,S)hhe_gradient_V_n_reduced(t,S), time, initial_condition, options);
figure;
plot(t, S(:,1));
xlabel('Time(ms)');
ylabel('Voltage (in mV)');
title('Response of V-n reduced HH model to 10uA/cm^2 current injection');
grid on;

% Response for current impulse. (includes values before/after threshold)
Iext = 0;
currents = linspace(0,15,5);
figure;
hold on;
xlabel('Time(ms)');
ylabel('Voltage (in mV)');
title('Simulating HH V-n Reduced model for current pulse');
time = [0 25];
for i = 1:5
    initial_condition = [-60+currents(i)/C, n_inf];
    [t, S] = ode15s(@(t,S)hhe_gradient_V_n_reduced(t,S),time, initial_condition, options);
    plot(t, S(:,1));
    hold on;
end
legend(strcat('impulse = ', num2str(currents(1))), strcat('impulse = ', num2str(currents(2))),strcat('impulse = ', num2str(currents(3))), strcat('impulse = ', num2str(currents(4))),strcat('impulse = ', num2str(currents(5))));
hold off;
%% Anode Break Excitation
Iext = -3;
hyperpol = 20;
time = [0 hyperpol];
% simulating 20 sec hyperpolarizing
initial_condition = [Vr, n_inf, m_inf, h_inf]; 
[t1, S1] = ode15s(@(t,S)hhe_gradient(t,S), time, initial_condition, options);

% removing Iext and resuming from where we left off
initial_condition = S1(end,:);
Iext = 0;
time = [0 100];
[t2,S2] = ode15s(@(t,S)hhe_gradient(t,S),time,initial_condition,options);
% putting the waveforms together
t = [t1;t2+hyperpol];
current_injection = [zeros(size(t1))-3;zeros(size(t2))];
V = [S1(:,1);S2(:,1)];
figure;
subplot(2,1,1);
plot(t,V);
xlabel("time in sec");
ylabel("membrane potential in mV");
title("Anode break excitation");
grid on;
subplot(2,1,2);
plot(t,current_injection);
xlabel("time in sec");
ylabel("current injection in uA/cm^2");
ylim([-10 10]);
grid on;
%% Phase plane analysis for V-m reduced model to explain anode break excitation
fprintf('\n------------------------- Part 18 ------------------------------ \n');
Vr = -60;    
alphan = @(V) -0.01 * (V + epsilon + 50)/(exp(-(V + epsilon + 50)/10)-1);
alpham = @(V) -0.1 * (V + epsilon + 35)/(exp(-(V + epsilon + 35)/10)-1);
alphah = @(V) 0.07 * exp(-(V + 60)/20);
betan = @(V) 0.125 * exp(-(V + 60)/80);
betam = @(V) 4 * exp(-(V + 60)/18);
betah = @(V) 1/(exp(-(V + 30)/10) + 1);

n_inf = alphan(Vr)/(alphan(Vr) + betan(Vr));
h_inf = alphah(Vr)/(alphah(Vr) + betah(Vr));

Iext = -3;
figure;
hold on;
grid on;
time_1 = [0 100];
initial_condition_1 = [Vr, m_inf];
[t1, S1] = ode15s(@(t,S)hhe_gradient_V_m_reduced(t,S,n_inf,h_inf), time_1, initial_condition_1, options);
V_null_1 = @(V) (((Iext - GK * (V -VK) * (n_inf^4)- GL*(V-VL))/(GNa*h_inf*(V-VNa)))^(1/3));
m_null_1 = @(V) alpham(V)/(alpham(V) + betam(V));

syms V m;
V_null = (1/C) * (Iext - GK * n_inf^4 * (V - VK) - GNa * m^3 * h_inf* (V - VNa) - GL * (V - VL)) == 0;
m_null = alpham *(1-m) - betam*m == 0 ;
soln1 = vpasolve([V_null, m_null], [V, m], [-60, 0.01]);
soln2 = vpasolve([V_null, m_null], [V, m], [50, 1]);
V_eq = double(soln1.V);
m_eq = double(soln1.m);
plot(V_eq, m_eq, 'o',"LineWidth",2);
text(V_eq+5, m_eq, ['(' num2str(round(V_eq,3)) ',' num2str(round(m_eq,3)) ')']);
V_eq = double(soln2.V);
m_eq = double(soln2.m);
plot(V_eq, m_eq, 'o',"LineWidth",2);
text(V_eq+5, m_eq-0.02, ['(' num2str(round(V_eq,3)) ',' num2str(round(m_eq,3)) ')']);
grid on;

syms V m;
am =  -0.1 * (V + epsilon + 35)/(exp(-(V + epsilon + 35)/10)-1);
bm = 4 * exp(-(V + 60)/18);
dV = (1/C) * (Iext - GK * n_inf^4 * (V - VK) - GNa * m^3 * h_inf* (V - VNa) - GL * (V - VL));
dm = am * (1 - m) - bm * m ;
J = jacobian([dV,dm],[V,m]);
V = double(soln1.V);
m = double(soln1.m);
J_mat = zeros(2,2);
J_mat(1,1) = subs(J(1,1));
J_mat(1,2) = subs(J(1,2));
J_mat(2,1) = subs(J(2,1));
J_mat(2,2) = subs(J(2,2));
eigenValues = eig(J_mat);
fprintf("\nStability analysis for Iext = -3 uA/cm^2\n");
fprintf('\nEquilibrium point %d : The eigen values are  %f%+fi , %f%+fi \n', 1, real(eigenValues(1)), imag(eigenValues(1)),real(eigenValues(2)), imag(eigenValues(2)));

if real(eigenValues(1)) < 0  && real(eigenValues(2)) < 0
    fprintf("\n ------------stable---------------\n");
else
    fprintf("\n ------------unstable---------------\n");
end


V = double(soln2.V);
m = double(soln2.m);
J_mat = zeros(2,2);
J_mat(1,1) = subs(J(1,1));
J_mat(1,2) = subs(J(1,2));
J_mat(2,1) = subs(J(2,1));
J_mat(2,2) = subs(J(2,2));
eigenValues = eig(J_mat);
fprintf('\nEquilibrium point %d : The eigen values are  %f%+fi , %f%+fi \n', 2, real(eigenValues(1)), imag(eigenValues(1)),real(eigenValues(2)), imag(eigenValues(2)));
if real(eigenValues(1)) < 0  && real(eigenValues(2)) < 0
    fprintf("\n ------------stable---------------\n");
else
    fprintf("\n ------------unstable---------------\n");
end


    

fplot(@(V) V_null_1(V), [-80 100], "LineWidth",1.5);
fplot(@(V) m_null_1(V), [-80 100],"LineWidth",1.5);
title("Phase plot for Iext = -3 uA/cm^2");
xlabel("V in mV");
ylabel("m");
plot(S1(:,1),S1(:,2),"LineWidth",1.5);
ylim([0 1.1]);
plot(Vr, m_inf, 'o',"LineWidth",2);
text(Vr+5, m_inf+0.02, ['(' num2str(round(Vr,3)) ',' num2str(round(m_inf,3)) ')']);
legend(["Eq. point: stable","Eq. point: stable","V null cline","m null cline","trajectory","initial conditions"],"Location","southeast");

Iext = 0;
figure;
hold on;
grid on;
time_2 = [0 50];
V_after_hyperpol = S1(end,1);
m_after_hyperpol = S1(end,2);
initial_condition_2 = [V_after_hyperpol,m_after_hyperpol];
n_inf = alphan(V_after_hyperpol)/(alphan(V_after_hyperpol) + betan(V_after_hyperpol));
h_inf = alphah(V_after_hyperpol)/(alphah(V_after_hyperpol) + betah(V_after_hyperpol));
[t2, S2] = ode15s(@(t,S)hhe_gradient_V_m_reduced(t,S,n_inf,h_inf), time_2, initial_condition_2, options);
V_null_2 = @(V) (((Iext - GK * (V -VK) * (n_inf^4)- GL*(V-VL))/(GNa*h_inf*(V-VNa)))^(1/3));
m_null_2 = @(V) alpham(V)/(alpham(V) + betam(V));

syms V m;
V_null = (1/C) * (Iext - GK * n_inf^4 * (V - VK) - GNa * m^3 * h_inf* (V - VNa) - GL * (V - VL)) == 0;
m_null = alpham *(1-m) - betam*m == 0 ;
soln3 = vpasolve([V_null, m_null], [V, m], [50, 1]);
V_eq = double(soln3.V);
m_eq = double(soln3.m);
plot(V_eq, m_eq, 'o',"LineWidth",2);
text(V_eq+5, m_eq-0.05, ['(' num2str(round(V_eq,3)) ',' num2str(round(m_eq,3)) ')']);
grid on;


syms V m;
am =  -0.1 * (V + epsilon + 35)/(exp(-(V + epsilon + 35)/10)-1);
bm = 4 * exp(-(V + 60)/18);
dV = (1/C) * (Iext - GK * n_inf^4 * (V - VK) - GNa * m^3 * h_inf* (V - VNa) - GL * (V - VL));
dm = am * (1 - m) - bm * m ;
J = jacobian([dV,dm],[V,m]);
V = double(soln3.V);
m = double(soln3.m);
J_mat = zeros(2,2);
J_mat(1,1) = subs(J(1,1));
J_mat(1,2) = subs(J(1,2));
J_mat(2,1) = subs(J(2,1));
J_mat(2,2) = subs(J(2,2));
eigenValues = eig(J_mat);
fprintf("\nStability analysis for Iext = 0 uA/cm^2\n");
fprintf('\nEquilibrium point %d : The eigen values are  %f%+fi , %f%+fi \n', 1, real(eigenValues(1)), imag(eigenValues(1)),real(eigenValues(2)), imag(eigenValues(2)));

if real(eigenValues(1)) < 0  && real(eigenValues(2)) < 0
    fprintf("\n ------------stable---------------\n");
else
    fprintf("\n ------------unstable---------------\n");
end



fplot(@(V) V_null_2(V), [-80 100], "LineWidth",1.5);
fplot(@(V) m_null_2(V), [-80 100],"LineWidth",1.5);
title("Phase plot for Iext = 0 uA/cm^2");
xlabel("V in mV");
ylabel("m");
plot(S2(:,1),S2(:,2),"LineWidth",1.5);
ylim([0 1.1]);
plot(V_after_hyperpol, m_after_hyperpol, 'o',"LineWidth",2);
text(V_after_hyperpol, m_after_hyperpol, ['(' num2str(round(Vr,3)) ',' num2str(round(m_inf,3)) ')']);
legend(["Eq. point: stable","V null cline","m null cline","trajectory","initial conditions"],"Location","southeast");
t_end = cputime;
fprintf("\nRun time is %0.2f minutes\n",(t_end-t_start)/60);
%% Thank you for your patience :)
%% MLE gradient
function dS = mle_gradient(~,S)

global C;
global GCa;
global VCa;
global GK;
global VK;
global GL;
global VL;
global V1;
global V2;
global V3;
global V4;
global phi;
global Iext;
%state variables:
V=S(1);
w=S(2);
%local functions:
m_inf = (0.5*(1+tanh((V-V1)/V2)));
w_inf = (0.5*(1+tanh((V-V3)/V4)));
tau_w_inv = cosh((V-V3)/(2*V4));
dV = (1/C)*(GCa*m_inf*(VCa-V) + GK*w*(VK-V) + GL*(VL-V)+Iext);
dw = phi*(w_inf-w)*tau_w_inv;
dS=[dV;dw];
end

%% HHE gradient
function dS = hhe_gradient(~,S)
global C;
global Iext;
global GK;
global GNa;
global GL;
global VK;
global VNa;
global VL;
global epsilon; 
V = S(1);
n = S(2);
m = S(3);
h = S(4);
alphan =  -0.01 * (V + epsilon + 50)/(exp(-(V + epsilon + 50)/10)-1);
alpham =  -0.1 * (V + epsilon + 35)/(exp(-(V + epsilon + 35)/10)-1);
alphah = 0.07 * exp(-(V + 60)/20);
betan = 0.125 * exp(-(V + 60)/80);
betam = 4 * exp(-(V + 60)/18);
betah = 1/(exp(-(V + 30)/10) + 1);

dV = (1/C) * (Iext - GK * n^4 * (V - VK) - GNa * m^3 * h * (V - VNa) - GL * (V - VL));
dn = alphan * (1 - n) - betan * n;
dm = alpham * (1 - m) - betam * m;
dh = alphah * (1 - h) - betah * h;  

dS = [dV; dn; dm; dh];
end
%% Spike Frequency Calculation
function f = spike_frequency(t,S)
V = S(:,1);
if max(V)<0
    f = 0;
    return;
end
action_count = 0;
start_idx = 1;
stop_idx = length(t);
flag = 1;
for i = 1:length(V)
    j = length(V)+1-i;
    if V(j) > 0
        flag = 0;
    end
    if V(j) < 0 && flag == 0
        stop_idx = j;
        break;
    end
end

flag = 1;
for i = 1:length(V)
    if V(i) < 0
        flag = 0;
    end
    if V(i) > 0 && flag == 0
        start_idx = i;
        break;
    end
end
flag = 0;
for i = start_idx:stop_idx
    if V(i) > 0 && flag == 0
        action_count = action_count + 1;
        flag = 1;
    end
    if V(i) < 0
        flag = 0;
    end
end
start_time = t(start_idx);
stop_time = t(stop_idx);
timespan = stop_time-start_time;
timespan = timespan/1000;
f = action_count/timespan;
end

%%
function stability_analysis(Iext)
global C;
global GK;
global GNa;
global GL;
global VK;
global VNa;
global VL;
global epsilon;

syms V n m h;
alphan =  -0.01 * (V + epsilon + 50)/(exp(-(V + epsilon + 50)/10)-1);
alpham =  -0.1 * (V + epsilon + 35)/(exp(-(V + epsilon + 35)/10)-1);
alphah = 0.07 * exp(-(V + 60)/20);
betan = 0.125 * exp(-(V + 60)/80);
betam = 4 * exp(-(V + 60)/18);
betah = 1/(exp(-(V + 30)/10) + 1);
V_null = (1/C) * (Iext - GK * n^4 * (V - VK) - GNa * m^3 * h* (V - VNa) - GL * (V - VL)) == 0;
n_null = alphan *(1-n) - betan*n == 0 ;
m_null = alpham * (1 - m) - betam * m ==0 ;
h_null = alphah * (1 - h) - betah * h ==0 ;
    
eq_pt = solve([V_null, n_null, m_null, h_null], [V, n, m ,h]);
V_eq1 = double(eq_pt.V);
n_eq1 = double(eq_pt.n);
m_eq1 = double(eq_pt.m);
h_eq1 = double(eq_pt.h);
    
for k=1:length(eq_pt)
    fprintf('\nEquilibrium Point %d :  V=%f n=%f m=%f h=%f\n',k, eq_pt(k).V,eq_pt(k).n,eq_pt(k).m,eq_pt(k).h);
end
% Stability Analysis
dV = (1/C) * (Iext - GK * n^4 * (V - VK) - GNa * m^3 * h* (V - VNa) - GL * (V - VL));
dn = alphan *(1-n) - betan*n;
dm = alpham * (1 - m) - betam * m ;
dh = alphah * (1 - h) - betah * h ;
    
J = jacobian([dV, dn, dm, dh],[V,n,m,h]);
V = V_eq1;
n = n_eq1;
m = m_eq1;
h = h_eq1;
J_mat = zeros(4,4);
for i = 1:4
    for j = 1:4
        J_mat(i,j) = subs(J(i,j));
    end
end
eigenValues = eig(J_mat);
fprintf('The eigen values are %f%+fi , %f%+fi , %f%+fi , %f%+fi \n', real(eigenValues(1)), imag(eigenValues(1)), ...
            real(eigenValues(2)), imag(eigenValues(2)), real(eigenValues(3)), imag(eigenValues(3)), ...
            real(eigenValues(4)), imag(eigenValues(4)));
k=0;
for x=1:4
    if real(eigenValues(x)) < 0 
        k = k+1;
    end
end
if k == 4
    fprintf("Stable\n");
else
    fprintf("Unstable\n");
end
end
%%
function dS = hhe_gradient_V_n_reduced(t,S)
    global C;
    global Iext;
    global GK;
    global GNa;
    global GL;
    global VK;
    global VNa;
    global VL;
    global epsilon;
    global h_inf_Vr;
    V = S(1);
    n = S(2);
    
    alphan =  -0.01 * (V + epsilon + 50)/(exp(-(V + epsilon + 50)/10)-1);
    alpham =  -0.1 * (V + epsilon + 35)/(exp(-(V + epsilon + 35)/10)-1);
    alphah = 0.07 * exp(-(V + 60)/20);
    betan = 0.125 * exp(-(V + 60)/80);
    betam = 4 * exp(-(V + 60)/18);
    betah = 1/(exp(-(V + 30)/10) + 1);
    
    m_inf = alpham/(alpham + betam);
    
    dV = (1/C)*(Iext - GNa* m_inf^3 * h_inf_Vr *(V-VNa) - GK* n^4 *(V - VK) - GL * (V - VL));
    dn = alphan * (1 - n) - betan * n;
    
    dS = [dV; dn];
end

%%
function dS = hhe_gradient_V_m_reduced(~,S,n_inf,h_inf)
global C;
global Iext;
global GK;
global GNa;
global GL;
global VK;
global VNa;
global VL;
global epsilon;    
V = S(1);
m = S(2);    
alpham =  -0.1 * (V + epsilon + 35)/(exp(-(V + epsilon + 35)/10)-1);
betam = 4 * exp(-(V + 60)/18);
    
dV = (1/C) * (Iext - GNa * m^3 * h_inf * (V - VNa)- GK * n_inf^4 * (V - VK) - GL * (V - VL));
dm = alpham * (1 - m) - betam * m;
dS = [dV; dm];
end