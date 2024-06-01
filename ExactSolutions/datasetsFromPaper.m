
clear; clc;

rho = 7.6e-6; %Density (kg/mm^3)
k = 0.025; %Thermal Conductivity (W/mm.K)
cp = 685.0; %Specific Heat Capacity (J/kg.K)
alpha = k/(rho*cp); %Thermal Diffusivity
T0 = 298.0; %Initial temperature (K)
T_ref = 298.0; %Temperature (K)
L = 100.0; %Length of rectangular domain (mm)
H = 50.0; %Width of rectangular domain (mm)

kx = 1/L;
kt = k/(rho*cp*L^2);

subDir = 'FromPaper';
cd(subDir)
mat_files = dir('*.mat');
cd ..
for i = 1:length(mat_files)
    
    file = mat_files(i).name;
    idx_ = find(file=='_');
    idxs = find(file=='s');
    t_str  = file(idx_(end)+1:idxs-1);
    r0_str = file(idx_(1)+1:idx_(end)-1);
    ts = str2double(t_str);
    if strcmp(r0_str, '002')
        r0 = 0.02;
    elseif strcmp(r0_str, '02')
        r0 = 0.2;
    elseif strcmp(r0_str, '2')
        r0 = 2;
    end
        
    load([subDir, '/', file])
    
%     idx = find(xyt(:, 2)==25);
%     xy_coarse  = xyt(1:idx(end), :);
%     T_coarse   = Tt(1:idx(end));
%     xy_overset = xyt(idx(end)+1:end, :);
%     T_overset  = Tt(idx(end)+1:end);
%     if any(ismember(xy_coarse, xy_overset, 'rows'))
%         asdasdsad
%     end
%     fig = figure(1);
%     scatter(xy_coarse(:, 1), xy_coarse(:, 2), 'o'); axis('equal'); hold on
%     scatter(xy_overset(:, 1), xy_overset(:, 2), '.r');
%     close(fig)
    
    idx = xyt(:, 2)==0;
    xyt_sym = xyt(~idx, :);
    xyt_sym(:, 2) = -xyt_sym(:, 2);
    Tt_sym = Tt(~idx);
    xyt_full = [xyt; xyt_sym];
    Tt_full = [Tt; Tt_sym];
    n = size(xyt_full, 1);
    
    nd_xy = xyt_full*kx;
    nd_ts = ts*kt*ones(n, 1);
    nd_r0 = r0*kx*ones(n, 1);
    XY = [nd_xy, nd_ts, nd_r0];
    T = Tt_full/T_ref;
    
    save(['ds' file(idx_(1):end)], 'XY', 'T')
    
    
end