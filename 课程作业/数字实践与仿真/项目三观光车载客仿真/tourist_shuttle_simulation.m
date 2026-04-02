function tourist_shuttle_simulation()
%==================== 离散事件仿真：景区观光车调度 ====================
clc; clear; rng(47); % 固定随机种子，便于复现

%% -------------------- 基本参数 --------------------
N = 5; % 观光车总数
capacity = 25; % 单车容量
load_spot = 3; % 同时装载车位数
up_time = 25; % 空车上山时间 (min)
down_time = 15; % 载客下山时间 (min)
max_load_time = 5; % 最大允许装载时间 (min)
lambda = 1/5; % 游客批次到达率 (批/min)
total_time = 120; % 仿真时段 16:00-18:00 (120 min)

% 团体规模分布
groupSizes = [1 2 3 4 5 6];
groupProbs = [0.05 0.25 0.30 0.15 0.15 0.10];

%% ------------------ 事件模板（统一字段） ------------------
eventTemplate = struct( ...
'time',NaN, 'type','', ...
'group_id',NaN, 'group_size',NaN, 'shuttle_id',NaN);

%% ------------------ 初始化系统状态 ------------------
eventQueue = []; % 事件队列（按时间升序）
shuttles = struct( ...
'id',(1:N)', ...
'status',repmat({'ready'},N,1), ...
'currentLoad',zeros(N,1), ...
'loadStartTime',zeros(N,1));
waitingGroups = []; % 等待上山的游客团体
loadingVec = []; % 正在装载的车辆 ID 列表
groupRecords = []; % 记录每个团体的到达/上车时间

%% ======================================================
% 生成 16:00-18:00 之间的游客到达事件
%% ======================================================
% 生成到达
[ats, szs] = genArrivals(lambda, total_time, groupSizes, groupProbs);

% 逐个插进事件队列
for i = 1:numel(ats)
    evt = eventTemplate;
    evt.time = ats(i);
    evt.type = 'arrival';
    evt.group_id = i;
    evt.group_size = szs(i);
    eventQueue = insertEvent(eventQueue, evt);
    
    % 同时初始化 groupRecords
    groupRecords(i).arrivalTime = ats(i);
    groupRecords(i).size = szs(i);
    groupRecords(i).boardingTime= NaN;
end

% 如果想立即看到达时间，加一行
disp('游客到达时间（分钟）:'); disp(ats);

% 初始车辆就绪事件
for id = 1:N
    evt = eventTemplate;
    evt.time = 0;
    evt.type = 'shuttle_ready';
    evt.shuttle_id = id;
    eventQueue = insertEvent(eventQueue,evt);
end

%% ======================================================
% 主事件循环
%% ======================================================
while ~isempty(eventQueue)
    [cur,eventQueue] = popEvent(eventQueue);
    now = cur.time;
    switch cur.type
        case 'arrival'
            % 加入等待队列
            waitingGroups(end+1).id = cur.group_id;
            waitingGroups(end).size = cur.group_size;
            waitingGroups(end).arrival= cur.time;
            % 尝试装载
            [eventQueue,loadingVec] = tryStartLoading(now,eventQueue,shuttles,...
            waitingGroups,loadingVec,load_spot,max_load_time);
        case 'shuttle_ready'
            id = cur.shuttle_id;
            shuttles(id).status = 'ready';
            [eventQueue,loadingVec] = tryStartLoading(now,eventQueue,shuttles,...
            waitingGroups,loadingVec,load_spot,max_load_time);
        case 'start_loading'
            id = cur.shuttle_id;
            shuttles(id).status = 'loading';
            shuttles(id).loadStartTime = now;
            % 安排 finish_loading
            evt = eventTemplate;
            evt.time = now + max_load_time;
            evt.type = 'finish_loading';
            evt.shuttle_id = id;
            eventQueue = insertEvent(eventQueue,evt);
        case 'finish_loading'
            id = cur.shuttle_id;
            % 按 FIFO 装满为止
            loaded = 0;
            rmIdx = [];
            for k = 1:numel(waitingGroups)
                if loaded + waitingGroups(k).size > capacity, break; end
                loaded = loaded + waitingGroups(k).size;
                groupRecords(waitingGroups(k).id).boardingTime = now;
                rmIdx(k) = k;
            end
            waitingGroups(rmIdx) = [];
            % 车辆状态变为 loaded
            shuttles(id).currentLoad = loaded;
            shuttles(id).status = 'loaded';
            loadingVec(loadingVec==id) = [];
            % 安排下山
            evt = eventTemplate;
            evt.time = now + down_time;
            evt.type = 'arrive_downhill';
            evt.shuttle_id = id;
            eventQueue = insertEvent(eventQueue,evt);
            % 看看能不能再开一条装载位
            [eventQueue,loadingVec] = tryStartLoading(now,eventQueue,shuttles,...
            waitingGroups,loadingVec,load_spot,max_load_time);
        case 'arrive_downhill'
            id = cur.shuttle_id;
            shuttles(id).status = 'downhill';
            shuttles(id).currentLoad= 0;
            % 安排空车返回
            evt = eventTemplate;
            evt.time = now + up_time;
            evt.type = 'return_uphill';
            evt.shuttle_id = id;
            eventQueue = insertEvent(eventQueue,evt);
        case 'return_uphill'
            id = cur.shuttle_id;
            shuttles(id).status = 'ready';
            % 车辆再次就绪
            evt = eventTemplate;
            evt.time = now;
            evt.type = 'shuttle_ready';
            evt.shuttle_id = id;
            eventQueue = insertEvent(eventQueue,evt);
    end
end

%% -------------------- 输出统计 --------------------
boarded = ~isnan([groupRecords.boardingTime]);
if any(boarded)
    wt = [groupRecords(boarded).boardingTime] - [groupRecords(boarded).arrivalTime];
    fprintf('\n========== 仿真结果 ==========\n');
    fprintf('总团体数: %d 已上车: %d\n',numel(groupRecords),sum(boarded));
    fprintf('平均等待: %.2f min\n',mean(wt));
    fprintf('最大等待: %.2f min\n',max(wt));
else
    fprintf('无游客完成上车\n');
end
end

%% ======================================================
% 工具函数
%% ======================================================
function q = insertEvent(q,e)
% 按时间升序插入事件结构体（字段已统一）
if isempty(q)
    q = e;
return;
end
t = [q.time];
idx = find(t>e.time,1,'first');
if isempty(idx)
    q(end+1) = e;
else 
    q = [q(1:idx-1),e,q(idx:end)];
end
end

function [e,q] = popEvent(q)
e = q(1); q = q(2:end);
end

function [eq,loadVec] = tryStartLoading(now,eq,shuttles,waitGroups,loadVec,maxLoad,~)
% 有空车位&有等待团体→触发 start_loading
while numel(loadVec)<3 && ~isempty(waitGroups)
    ready = find(strcmp({shuttles.status},'ready'),1);
    if isempty(ready), break; end
    evt = struct('time',now,'type','start_loading','shuttle_id',ready,...
    'group_id',NaN,'group_size',NaN);
    eq = insertEvent(eq,evt);
    loadVec(end+1) = ready; % 加入装载列表
end
end

function [ats, szs] = genArrivals(lam, T, groupSizes, groupProbs)
% 返回两个等长向量：ats=到达时刻，szs=团体人数
ats = []; szs = [];
t = 0;
while t <= T
    dt = -log(rand)/lam; % 指数间隔
    t = t + dt;
    if t > T, break; end % 严格只保留 [0,T] 内
    ats(end+1) = t;
    szs(end+1) = randsample(groupSizes,1,true,groupProbs);
end
end