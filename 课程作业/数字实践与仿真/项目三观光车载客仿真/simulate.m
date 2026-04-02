function [avg_wait_time,max_wait_time,waste_rate]=simulate(N,capacity)
%备注：假设上下山时间固定且不考虑上山的游客，装载时间为均匀分布U(3,5)，装载时若有空位到达的游客也可上车，装载时间结束无论是否满载都发车
% 核心参数设置
lambda = 60 / 5;          % 泊松到达率（批/小时），每5分钟一批
total_time = 60;          % 仿真总时间（分钟，16:00-17:00）
loading_time = @() 3 + (5-3)*rand;  % 装载时间（均匀分布 U(3,5) 分钟）
up_time = 25;             % 上山时间（分钟）
down_time = 15;           % 下山时间（分钟）
loading_spots = 3;        % 同时装载车位数量

% 状态变量初始化
clock = 0;                % 仿真时钟
event_queue = [];         % 事件队列（时间戳+类型）
queue_length = 0;         % 游客队列长度
vehicles_ready = N;       % 待命车辆数
busy_spots = 0;           % 占用装载位数量
wait_times = [];          % 游客等待时间记录
total_seats = 0;          % 按车次计的总座位数
waste_seats = 0;          % 没坐人浪费的总座位数
spare_seats = 0;          % 正在装载的车的剩余座位数
queue_arrival = [];

% 生成游客到达事件
arrival_times = [];
current_time = 0;
while current_time <= total_time
    inter_arrival = -log(rand) / lambda;  % 指数分布间隔时间
    current_time = current_time + inter_arrival;
    if current_time <= total_time
        arrival_times = [arrival_times, current_time];
        % 添加到达事件到队列（事件类型：'arrival'）
        event_queue = [event_queue, struct('time', current_time, 'type', 'arrival','start_time',0)];
    end
end

while ~isempty(event_queue)
    % 按时间戳排序，提取下一个事件
    [min_time, idx] = min([event_queue.time]);
    current_event = event_queue(idx);
    clock = min_time;
    event_queue(idx) = [];  % 移除已处理事件

    % 处理事件类型
    switch current_event.type
        case 'arrival'  % 游客到达
            queue_length = queue_length + 1;  % 队列+1
            queue_arrival = [queue_arrival,current_event.time]; % 记录排队到达时间
            %如果排队人数多于可用座位数
            if queue_length > spare_seats
                %尝试使用空闲车开始装载
                if vehicles_ready > 0 && busy_spots < loading_spots
                    busy_spots = busy_spots + 1;
                    vehicles_ready = vehicles_ready - 1;
                    start_load_time = clock;
                    finish_load_time = clock + loading_time();
                    total_seats = total_seats + capacity;
                    spare_seats = spare_seats + capacity;
                    %按队列上车
                    while (~isempty(queue_arrival)) && spare_seats
                        wait_times=[wait_times,current_event.time-queue_arrival(1)];
                        queue_arrival(1)=[];
                        queue_length=queue_length-1;
                        spare_seats=spare_seats-1;
                    end
                    % 添加装载完成事件
                    event_queue = [event_queue, struct('time', finish_load_time, 'type', 'finish_loading', 'start_time', start_load_time)];
                end
            %如果排队人数小于空闲座位数
            else
                %按队列上车
                while (~isempty(queue_arrival)) && spare_seats
                    wait_times=[wait_times,current_event.time-queue_arrival(1)];
                    queue_arrival(1)=[];
                    queue_length=queue_length-1;
                    spare_seats=spare_seats-1;
                end
            end

        case 'finish_loading'  % 装载完成，车辆发车
            %如果只有自己一辆车正在装载，发车后应该将spare_seats设为0
            if busy_spots==1
                waste_seats=waste_seats+spare_seats;
                spare_seats=0;
            end
            busy_spots = busy_spots - 1;
            % 车辆发车后，添加返回事件（上山+下山时间）
            return_time = current_event.time + up_time + down_time;
            event_queue = [event_queue, struct('time', return_time, 'type', 'vehicle_return','start_time',0)];

        case 'vehicle_return'  % 车辆返回停车场
            vehicles_ready = vehicles_ready + 1;
            % 若有等待游客，立即开始新一轮装载
            if queue_length > spare_seats
                %尝试使用空闲车开始装载
                if vehicles_ready > 0 && busy_spots < loading_spots
                    busy_spots = busy_spots + 1;
                    vehicles_ready = vehicles_ready - 1;
                    start_load_time = clock;
                    finish_load_time = clock + loading_time();
                    total_seats = total_seats + capacity;
                    spare_seats = spare_seats + capacity;
                    %按队列上车
                    while (~isempty(queue_arrival)) && spare_seats
                        wait_times=[wait_times,current_event.time-queue_arrival(1)];
                        queue_arrival(1)=[];
                        queue_length=queue_length-1;
                        spare_seats=spare_seats-1;
                    end
                    % 添加装载完成事件
                    event_queue = [event_queue, struct('time', finish_load_time, 'type', 'finish_loading', 'start_time', start_load_time)];
                end
            %如果排队人数小于空闲座位数
            else
                %按队列上车
                while (~isempty(queue_arrival)) && spare_seats
                    wait_times=[wait_times,current_event.time-queue_arrival(1)];
                    queue_arrival(1)=[];
                    queue_length=queue_length-1;
                    spare_seats=spare_seats-1;
                end
            end
    end
end
avg_wait_time=mean(wait_times);
max_wait_time=max(wait_times);
waste_rate=(waste_seats/total_seats)*100;