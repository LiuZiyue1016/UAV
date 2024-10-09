import time
import numpy as np
import airsim
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2

def image_test(target_number, target_circle, target_judge, model, img):
    result = inference_detector(model, img)
    # show_result_pyplot(model, img, result)
    # 获取所有检测到的对象的类别和置信度
    class_names = model.CLASSES
    obj_list = []

    if all(bboxes.size > 0 for bboxes in result):
        max_confidence = max([bbox[-1] for bboxes in result for bbox in bboxes])
    else:
        max_confidence = 0.0  # 当序列为空时，将最大置信度设为 0.0

    # 置信度大于0.6时则成功
    for i, bboxes in enumerate(result):
        obj = []
        for bbox in bboxes:
            if bbox[-1] > 0.8 and max_confidence != 0.0:
                obj.append({'label': class_names[i], 'bbox': bbox[:-1], 'confidence': bbox[-1] / max_confidence})
            elif bbox[-1] > 0.8 and max_confidence == 0.0:
                obj.append({'label': class_names[i], 'bbox': bbox[:-1], 'confidence': 0.0})
        obj_list.append(obj)

    # 初始化数字1、2、3和circle的列表
    numbers = {1: [], 2: [], 3: []}
    circle = []

    # 遍历检测结果
    for i, obj in enumerate(obj_list):
        # 判断当前目标框的类别
        if i == 0 or i == 1 or i == 2:
            # 将数字1、2、3的目标框添加到对应的数字列表中
            if obj and obj[0].get('label') and obj[0]['label'] in ('1', '2', '3'):
                label = int(obj[0]['label'])  # 获取数字1、2、3的标签
                numbers[label] += obj
        elif i == 3:
            # 将circle的目标框添加到circle列表中
            circle += obj

    # 遍历所有的circle目标框，将其和数字1、2、3对应起来
    for circle_obj in circle:
        cx1, _, cx2, _ = circle_obj['bbox']
        for label, target_list in numbers.items():
            for target in target_list:
                x1, _, x2, _ = target['bbox']
                if abs(cx1 - x1) < 50 and abs(cx2 - x2) < 50:
                    target['circle'] = circle_obj  # 将对应的circle目标框添加到数字目标框的属性中

    # 打印数字1、2、3和circle的目标框，输出识别的参数和结果
    for label, target_list in numbers.items():
        for i, target in enumerate(target_list):
            circle_label = target['circle']['label'] if 'circle' in target else 'None'
            if target_number == target['label']:
                target_judge = 'success'
                target_circle = target['circle']['bbox'] if 'circle' in target else 'None'
        end = [target_judge, target_circle]
    return end

def get_water(client):
    # 中间点和终点位置
    midpoint = np.array([2, 35.5, -10])
    endpoint = np.array([2, 38.5, -6])

    # 当前目标位置
    target = midpoint

    # 记录位置
    x = []
    y = []
    z = []

    # 控制循环
    while True:
        # 获取无人机当前位置
        pos = client.getMultirotorState().kinematics_estimated.position
        current = np.array([pos.x_val, pos.y_val, pos.z_val])

        # 计算位置误差
        error = target - current
        # print("Position: ", "x=", pos.x_val, "y=", pos.y_val, "z=", pos.z_val, "error:", error)

        # 记录位置
        x.append(current[0])
        y.append(current[1])
        z.append(current[2])

        # 检查是否到达目标位置
        if error[1] < 1:
            if (target == midpoint).all():
                target = endpoint
            else:
                break
        # 控制无人机运动
        client.moveToPositionAsync(target[0], target[1], target[2], 5, 0.1).join()

def through_circle(client, target_number, target_circle, target_judge, model):
    d = 1
    s = 0
    # 获取无人机当前位置
    pos = client.getMultirotorState().kinematics_estimated.position
    while True:
         # 拍摄并识别正前方的图像，
         responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene)])
         img_png = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)  # frombuffer将data以流的形式读入转化成ndarray对象
         img_bgr = cv2.imdecode(img_png, cv2.IMREAD_COLOR)  # 从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式;
         target_judge = image_test(target_number, [], 'fail', model, img_bgr)[0]

         # 判断能否正确识别到目标数字
         if target_judge == 'fail':
             print("已距离圆环足够近，请穿过后就停止")
             # 获取无人机当前位置
             pos = client.getMultirotorState().kinematics_estimated.position
             if target_number == '1':
                 s = 9.5
             elif target_number == '2':
                 s = 4.5
             elif target_number == '3':
                 s = 7.5
             # 由于识别精度不同，前进距离不同
             client.moveToPositionAsync(pos.x_val, pos.y_val+s, -13, 1).join()
             break
         # 移动
         client.moveToPositionAsync(pos.x_val, pos.y_val+d, -13, 1)
         d += 1
    return

def move_back(client):
    # PID参数
    kp = 1.5
    ki = 0.1
    kd = 1.2

    # 定义目标位置
    target_position = np.array([0.0, 0.0, -2.0])

    # 时间间隔
    dt = 0.05

    # 运行时间
    run_time = 12

    # 最大速度和加速度
    max_vel = 10.0
    max_acc = 5.0

    # 获取初始状态
    pos = client.getMultirotorState().kinematics_estimated.position
    current_position = np.array([pos.x_val, pos.y_val, pos.z_val])

    error_sum = 0
    last_error = 0
    control = np.zeros(3)

    while True:
        # 计算误差
        pos = client.getMultirotorState().kinematics_estimated.position
        current_position = np.array([pos.x_val, pos.y_val, pos.z_val])
        error = target_position - current_position

        if pos.z_val > -2.0:
            # 无人机悬停3秒钟
            client.hoverAsync().join()
            time.sleep(1)
        # 计算PID控制量
        p_term = kp * error
        i_term = ki * (error_sum + error * dt)
        d_term = kd * ((error - last_error) / dt)

        # 计算总控制量
        control = p_term + i_term + d_term

        # 限制控制量
        control_norm = np.linalg.norm(control)
        if control_norm > max_acc:
            control = control / control_norm * max_acc
        v = client.getMultirotorState().kinematics_estimated.linear_velocity
        v_arr = np.array([v.x_val, v.y_val, v.z_val])
        vel_norm = np.linalg.norm(v_arr)
        if vel_norm > max_vel:
            control = control - (vel_norm - max_vel) * control / control_norm

        # 更新状态
        client.moveByVelocityAsync(control[0], control[1], control[2], dt).join()
        error_sum += error
        last_error = error

        # 输出当前位置和控制量
        # print(f"Current Position: {current_position}, Control: {control}")

        # 模拟飞行器运动延迟
        time.sleep(dt)

        diatance = np.linalg.norm(current_position - target_position)
        if diatance < 2:
            break

    # 无人机悬停3秒钟
    client.hoverAsync().join()
    time.sleep(1)
    client.moveToPositionAsync(0.0, 0.0, -2.0, 1).join()

    return

# 设置目标信息
target_number = '3'
target_circle = []
target_judge = 'fail'

# 连接无人机并起飞
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync()

#定义视觉识别文件
config_file = 'checkpoint/faster_rcnn_r50_fpn_2x_coco.py'
checkpoint_file = 'checkpoint/latest.pth'
device = 'cuda:0'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)

# 飞到水库上方并接水
get_water(client)
print("已成功接水，请穿过目标数字下的圆环处灭火")

# 在运动中视觉识别，对准目标数字下的圆环， 穿越圆环灭火
x = 0
v = 1
client.moveToPositionAsync(2, 42, -13, v, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                           yaw_mode=airsim.YawMode(True, 0)).join()
while True:
    # 拍摄正前方的图像
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene)])
    img_png = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)  # frombuffer将data以流的形式读入转化成ndarray对象
    img_bgr = cv2.imdecode(img_png, cv2.IMREAD_COLOR)  # 从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式;
    # 识别图像
    target_circle = image_test(target_number, [], 'fail', model, img_bgr)[1]
    # 判断当前图像中是否有目标数字
    if any(target_circle) == True:
        # 根据目标数字下圆环中心点的坐标判断是否对准
        center = int(target_circle[0] + target_circle[2])/2
        if center >= 600 and center <= 680:# 当到达一定范围时加速
            v = 0.5
        if center >= 630 and center <= 650:
            print(f"已对准目标数字 {target_number} 下的圆环， 接下来将向前行驶")
            through_circle(client, target_number, target_circle, target_judge, model)
            break
        if center > 650:
            x += 0.5
        elif center < 630:
            x -= 0.5
    else:
        print(f"当前的图像中没有目标数字 {target_number} ，请控制无人机进行移动后再识别")
        x += 0.5
    client.moveToPositionAsync(2-x, 42, -13, v, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                               yaw_mode=airsim.YawMode(True, 0))
print("已成功灭火，请回到起降区白色圆环内")

# 无人机悬停1秒钟
client.hoverAsync().join()
time.sleep(1)

# 返回起降区白色圆环内
client.moveByVelocityAsync(0, 0, -3, 3).join()
move_back(client)
print("已处于白色圆圈内，请降落")

# 降落并释放控制
client.landAsync().join()  # land
client.armDisarm(False)  # lock
client.enableApiControl(False)  # release control
