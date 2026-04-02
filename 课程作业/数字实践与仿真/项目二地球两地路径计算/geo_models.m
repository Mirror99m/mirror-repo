function geo_models(action)
%GEO_MODELS 完整测试：任务1~3一键运行
if nargin<1, action='test'; end
if strcmp(action,'test')
    %% ---------- 文档固定坐标 ----------
    latCD = 30.6558;  lonCD = 104.0656;   % 成都天府广场
    latXA = 34.2586;  lonXA = 108.9389;   % 西安钟楼

    %% ---------- 模型实例 ----------
    sph = earth_model_sphere();
    wgs = earth_model_wgs84();

    %% ---------- 任务1 向东100 km ----------
    [lat1_s, lon1_s] = destination_point(latCD, lonCD, 90, 100e3, 100, sph);
    [lat1_w, lon1_w] = destination_point(latCD, lonCD, 90, 100e3, 100, wgs);
    fprintf('\n=== 任务1 向东100 km ===\n');
    fprintf('球面模型终点：%.6f°N, %.6f°E\n', lat1_s, lon1_s);
    fprintf('WGS-84   终点：%.6f°N, %.6f°E\n', lat1_w, lon1_w);

    %% ---------- 任务2 北偏东30° 200 km ----------
    [lat2_s, lon2_s] = destination_point(latCD, lonCD, 30, 200e3, 200, sph);
    [lat2_w, lon2_w] = destination_point(latCD, lonCD, 30, 200e3, 200, wgs);
    fprintf('\n=== 任务2 北偏东30° 200 km ===\n');
    fprintf('球面模型终点：%.6f°N, %.6f°E\n', lat2_s, lon2_s);
    fprintf('WGS-84   终点：%.6f°N, %.6f°E\n', lat2_w, lon2_w);

    %% ---------- 任务3 成都→西安最短路径 ----------
    [d_sph, az_sph] = haversine_inv(latCD, lonCD, latXA, lonXA);
    [d_wgs, az_wgs] = wgs84_inv  (latCD, lonCD, latXA, lonXA);
    n = 100;                                    % 100段
    [latPath, lonPath] = great_circle_path(latCD, lonCD, latXA, lonXA, n);
    distPath = linspace(0, d_sph, n)';          % 累计距离
    azPath   = repmat(az_sph, n, 1);

    fprintf('\n=== 任务3 成都→西安最短路径 ===\n');
    fprintf('球面距离：%.3f km，初始方位角：%.2f°\n', d_sph, az_sph);
    fprintf('WGS-84距离：%.3f km，初始方位角：%.2f°\n', d_wgs, az_wgs);

    %% 5. 写CSV（当前文件夹 points.csv）
    pointNames = cellstr(sprintfc('P%03d',1:n)');   % ← 关键修复
    T = table(pointNames, latPath, lonPath, distPath, azPath,...
             'VariableNames',{'Point_Name','Latitude','Longitude',...
                              'Distance_km','Azimuth_deg'});
    writetable(T, fullfile(pwd,'points.csv'));
    fprintf('已输出 points.csv（%d 行）到当前文件夹\n', n);
end
end

%% ---------------- 地球模型 ----------------
function model = earth_model_sphere()
model.name='sphere';  model.a=6371000;  model.f=0;  model.b=model.a;  model.e2=0;
end
function model = earth_model_wgs84()
model.name='wgs84';   model.a=6378137;  model.f=1/298.257223563;
model.b=model.a*(1-model.f);  model.e2=2*model.f-model.f^2;
end

%% ---------------- 前向问题（含nstep） ----------------
function [lat2,lon2] = destination_point(lat1,lon1,theta,total_m,nstep,model)
if nstep<=0, nstep=1; end
step_m = total_m / nstep;              % 每段长度（米）
switch model.name
    case 'sphere'
        R = model.a;
        delta = step_m / R;            % 每段角距离
        lat = lat1*pi/180;  lon = lon1*pi/180;  az = theta*pi/180;
        for k=1:nstep
            lat_new = asin(sin(lat)*cos(delta) + cos(lat)*sin(delta)*cos(az));
            lon_new = lon + atan2(sin(az)*sin(delta)*cos(lat), ...
                                  cos(delta)-sin(lat)*sin(lat_new));
            lat = lat_new;  lon = lon_new;
        end
        lat2 = lat*180/pi;  lon2 = lon*180/pi;
        lon2 = mod(lon2+540,360)-180;
    case 'wgs84'
        [X,Y,Z] = ll2ecef(lat1,lon1,0,model);
        R = enu_rotmat(lat1,lon1);
        az = theta*pi/180;
        for k=1:nstep
            dENU = [step_m*sin(az); step_m*cos(az); 0];
            dXYZ = R*dENU;
            X=X+dXYZ(1); Y=Y+dXYZ(2); Z=Z+dXYZ(3);
            [lat1,lon1] = ecef2ll(X,Y,Z,model);
            R = enu_rotmat(lat1,lon1);
        end
        lat2=lat1; lon2=lon1;
end
end

%% ---------------- 反向问题 ----------------
function [d_km,az12]=haversine_inv(lat1,lon1,lat2,lon2)
lat1=lat1*pi/180; lon1=lon1*pi/180; lat2=lat2*pi/180; lon2=lon2*pi/180;
dlat=lat2-lat1; dlon=lon2-lon1;
a=sin(dlat/2)^2+cos(lat1)*cos(lat2)*sin(dlon/2)^2;
c=2*atan2(sqrt(a),sqrt(1-a));
d_km=6371*c;
y=sin(dlon)*cos(lat2); x=cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(dlon);
az12=atan2(y,x)*180/pi;
end
function [d_km,az12]=wgs84_inv(lat1,lon1,lat2,lon2)
[d_km,az12]=haversine_inv(lat1,lon1,lat2,lon2);
d_km=d_km*1.0005;   % 简易缩放
end

%% ---------------- 大圆路径离散 ----------------
function [latPath,lonPath]=great_circle_path(lat1,lon1,lat2,lon2,n)
lat1=lat1*pi/180; lon1=lon1*pi/180; lat2=lat2*pi/180; lon2=lon2*pi/180;
dlat=lat2-lat1; dlon=lon2-lon1;
a=sin(dlat/2)^2+cos(lat1)*cos(lat2)*sin(dlon/2)^2;
c=2*atan2(sqrt(a),sqrt(1-a));
f=linspace(0,1,n)';
A=sin((1-f)*c)./sin(c); B=sin(f*c)./sin(c);
x=A*cos(lat1)*cos(lon1)+B*cos(lat2)*cos(lon2);
y=A*cos(lat1)*sin(lon1)+B*cos(lat2)*sin(lon2);
z=A*sin(lat1)+B*sin(lat2);
latPath=atan2(z,sqrt(x.^2+y.^2))*180/pi;
lonPath=atan2(y,x)*180/pi;
end

%% ---------------- 工具库 ----------------
function [X,Y,Z]=ll2ecef(lat,lon,h,model)
lat=lat*pi/180; lon=lon*pi/180;
N=model.a./sqrt(1-model.e2*sin(lat).^2);
X=(N+h).*cos(lat).*cos(lon);
Y=(N+h).*cos(lat).*sin(lon);
Z=(N*(1-model.e2)+h).*sin(lat);
end
function [lat,lon]=ecef2ll(X,Y,Z,model)
lon=atan2(Y,X); p=sqrt(X.^2+Y.^2); e2=model.e2;
lat=atan(Z./(p*(1-e2)));
for k=1:5
    N=model.a./sqrt(1-e2*sin(lat).^2);
    lat_new=atan((Z+e2*N.*sin(lat))./p);
    if max(abs(lat_new-lat))<1e-12, break; end
    lat=lat_new;
end
lat=lat*180/pi; lon=lon*180/pi;
end
function R=enu_rotmat(lat,lon)
lat=lat*pi/180; lon=lon*pi/180;
R=[-sin(lon), cos(lon), 0;
   -sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat);
    cos(lat)*cos(lon),  cos(lat)*sin(lon), sin(lat)]';
end