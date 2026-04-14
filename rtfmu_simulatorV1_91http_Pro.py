import time
import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from collections import deque
import threading
import datetime
import json
import re
import os
import asyncio
import websockets
import concurrent.futures
import logging
from concurrent.futures import ThreadPoolExecutor

class RTFmuSimulator:
    """
    基于PyFMI和Co-Simulation FMU的实时仿真器 (版本1.93)
    支持状态保存和恢复，便于在任意时间点暂停与继续
    增强了变量监控和控制功能
    新增WebSocket远程控制和监控接口
    支持模型参数辅助识别和诊断
    新增参数名筛查功能
    """
    
    def __init__(self, fmu_path, variables_to_monitor=None, step_size=1.0, real_time_factor=1.0):
        """
        初始化实时仿真器
        
        参数:
            fmu_path (str): FMU文件路径
            variables_to_monitor (list): 要监控的变量列表，默认为None(自动检测)
            step_size (float): 仿真步长，默认1.0秒
            real_time_factor (float): 实时因子，>1表示加速，<1表示减速
        """
        # 加载FMU模型(以Co-Simulation模式)
        self.model = load_fmu(fmu_path, kind='cs')
        self.model_name = self.model.get_name()
        print(f"已加载模型: {self.model_name}")
        
        # 检查是否支持FMU状态获取/设置
        try:
            state = self.model.get_fmu_state()
            self.model.set_fmu_state(state)
            self.supports_state = True
            print("✓ 模型支持FMU状态获取/设置")
        except:
            self.supports_state = False
            print("✗ 模型不支持FMU状态获取/设置，部分功能将不可用")
        
        # 设置仿真参数
        self.step_size = step_size
        self.real_time_factor = real_time_factor
        
        # 设置变量监控
        if variables_to_monitor is None:
            # 如果未指定，尝试自动检测一些关键变量
            all_vars = self.model.get_model_variables()
            self.variables = []
            # 优先选择输出变量和一些看起来重要的变量
            for var in all_vars:
                causality = all_vars[var].causality
                if causality == 3:  # 3 表示 output
                    self.variables.append(var)
                elif 'out' in var.lower() or 'state' in var.lower() or 'result' in var.lower():
                    self.variables.append(var)
                # 限制变量数
                if len(self.variables) >= 10:
                    break
        else:
            self.variables = variables_to_monitor
            
        print(f"监控变量: {', '.join(self.variables)}")
        
        # 仿真状态
        self.current_time = 0.0
        self.initialized = False
        self.running = False
        self.paused = False
        
        # 结果存储
        self.results = {
            'time': deque(maxlen=10000)  # 限制存储点数以避免内存过度使用
        }
        for var in self.variables:
            self.results[var] = deque(maxlen=10000)
        
        # 状态保存
        self.saved_states = {}  # 格式: {时间点: FMU状态}
        
        # 实时仿真控制
        self.rt_thread = None
        self.stop_flag = threading.Event()
        
        # 变量控制
        self.variable_overrides = {}  # 需要覆盖的变量值
        self.modifiable_variables = []  # 可修改的变量列表
        self.scheduled_changes = []  # 调度的变量更改 [(变量名, 值, 时间点), ...]
        self.variable_lock = threading.Lock()  # 变量访问锁
        
        # 性能监控
        self.perf_metrics = {
            'step_time': deque(maxlen=100),
            'sync_time': deque(maxlen=100),
            'total_steps': 0,
            'overruns': 0
        }
        
        # WebSocket服务
        self.ws_server = None
        self.ws_thread = None
        self.ws_clients = set()
        self.ws_running = False
        self.ws_lock = threading.Lock()
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # 日志
        self.logger = logging.getLogger("RTFmuSimulator")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # 添加HTTP服务器
        self.http_server = None
    
    def initialize(self, start_time=0.0, variables=None, modifiable_variables=None):
        """
        初始化模型和仿真环境
        
        参数:
            start_time (float): 起始时间
            variables (list): 要监控的变量列表
            modifiable_variables (list): 可修改的变量列表
        """
        try:
            # 更新监控变量（如果提供）
            if variables is not None:
                self.variables = variables
                # 重新初始化结果存储
                self.results = {'time': deque(maxlen=10000)}
                for var in self.variables:
                    self.results[var] = deque(maxlen=10000)
            
            # 更新可修改变量（如果提供）
            if modifiable_variables is not None:
                self.modifiable_variables = modifiable_variables
            else:
                # 自动检测可修改变量
                self._detect_modifiable_variables()
            
            # 设置起始时间
            self.current_time = start_time
            
            # 初始化模型
            self.model.initialize(start_time)
            
            # 如果支持状态保存，保存初始状态
            if self.supports_state:
                self.saved_states[start_time] = self.model.get_fmu_state()
            
            # 存储初始值
            self._store_results()
            
            self.initialized = True
            self.logger.info(f"模型已初始化，起始时间: {start_time}")
            
            return True
        except Exception as e:
            self.logger.error(f"初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_modifiable_variables(self):
        """自动检测可修改变量 (内部方法)"""
        all_vars = self.model.get_model_variables()
        self.modifiable_variables = []
        
        for var in all_vars:
            try:
                causality = all_vars[var].causality
                if causality == 1 or causality == 2:  # 1=input, 2=parameter
                    self.modifiable_variables.append(var)
            except:
                pass
        
        self.logger.info(f"已检测到 {len(self.modifiable_variables)} 个可修改变量")
    
    def step(self):
        """
        执行单步仿真
        
        返回:
            bool: 成功返回True，失败返回False
        """
        if not self.initialized:
            self.logger.error("模型尚未初始化")
            return False
        
        try:
            step_start = time.time()
            
            # 应用调度的变量更改
            self._apply_scheduled_changes()
            
            # 应用变量覆盖
            with self.variable_lock:
                for var, value in self.variable_overrides.items():
                    self.model.set(var, value)
            
            # 执行仿真步骤
            self.model.do_step(current_t=self.current_time, step_size=self.step_size)
            
            # 更新当前时间
            self.current_time += self.step_size
            
            # 保存当前状态
            if self.supports_state:
                self.saved_states[self.current_time] = self.model.get_fmu_state()
                # 限制保存的状态数量
                if len(self.saved_states) > 100:
                    oldest_time = min(self.saved_states.keys())
                    del self.saved_states[oldest_time]
            
            # 存储结果
            self._store_results()
            
            # 计算并存储步骤执行时间
            step_time = time.time() - step_start
            self.perf_metrics['step_time'].append(step_time)
            self.perf_metrics['total_steps'] += 1
            
            # 检查是否有性能过载
            if step_time > (self.step_size / self.real_time_factor):
                self.perf_metrics['overruns'] += 1
            
            # 广播结果（如果有WebSocket客户端）
            if self.ws_clients:
                self._broadcast_results()
            
            return True
        except Exception as e:
            self.logger.error(f"执行步骤失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _store_results(self):
        """存储当前变量值 (内部方法)"""
        self.results['time'].append(self.current_time)
        
        for var in self.variables:
            try:
                value = self.model.get(var)
                self.results[var].append(value)
            except Exception as e:
                self.logger.warning(f"读取变量 {var} 失败: {str(e)}")
                self.results[var].append(None)
    
    def add_modifiable_variable(self, variable_name):
        """
        添加可修改的变量
        
        参数:
            variable_name (str): 变量名
            
        返回:
            bool: 成功返回True
        """
        if variable_name not in self.modifiable_variables:
            self.modifiable_variables.append(variable_name)
        return True
    
    def set_variable(self, variable_name, value):
        """
        设置变量值
        
        参数:
            variable_name (str): 变量名
            value: 要设置的值
            
        返回:
            bool: 成功返回True，失败返回False
        """
        try:
            # 检查变量是否可修改
            if variable_name not in self.modifiable_variables:
                self.logger.warning(f"变量 {variable_name} 不在可修改列表中")
                # 尝试添加到可修改列表
                self.add_modifiable_variable(variable_name)
            
            # 如果仿真未运行，直接修改
            if not self.running:
                self.model.set(variable_name, value)
                self.logger.info(f"变量已设置: {variable_name} = {value}")
                return True
            
            # 否则添加到覆盖列表
            with self.variable_lock:
                self.variable_overrides[variable_name] = value
            
            self.logger.info(f"变量将在下一步仿真时更改: {variable_name} = {value}")
            return True
        except Exception as e:
            self.logger.error(f"设置变量 {variable_name} 失败: {str(e)}")
            return False
    
    def schedule_variable_change(self, variable_name, value, at_time):
        """
        调度在指定时间修改变量
        
        参数:
            variable_name (str): 变量名
            value: 要设置的值
            at_time (float): 修改时间点
            
        返回:
            bool: 成功返回True，失败返回False
        """
        try:
            # 检查时间是否有效
            if at_time < self.current_time:
                self.logger.error(f"无法调度到过去的时间点: {at_time} < {self.current_time}")
                return False
            
            # 检查变量是否可修改
            if variable_name not in self.modifiable_variables:
                self.logger.warning(f"变量 {variable_name} 不在可修改列表中")
                # 尝试添加到可修改列表
                self.add_modifiable_variable(variable_name)
            
            # 添加到调度列表
            with self.variable_lock:
                self.scheduled_changes.append((variable_name, value, at_time))
                # 按时间排序
                self.scheduled_changes.sort(key=lambda x: x[2])
            
            self.logger.info(f"已调度变量更改: {variable_name} = {value} @ {at_time}s")
            return True
        except Exception as e:
            self.logger.error(f"调度变量更改失败: {str(e)}")
            return False
    
    def _apply_scheduled_changes(self):
        """应用调度的变量更改(内部方法)"""
        with self.variable_lock:
            to_remove = []
            
            for i, (var_name, value, at_time) in enumerate(self.scheduled_changes):
                if at_time <= self.current_time:
                    try:
                        self.model.set(var_name, value)
                        self.logger.info(f"已应用调度的变量更改: {var_name} = {value} @ {at_time}s")
                        to_remove.append(i)
                    except Exception as e:
                        self.logger.error(f"应用调度的变量更改失败: {var_name} = {value}: {str(e)}")
                        to_remove.append(i)
            
            # 移除已处理的条目 (从后向前移除，防止索引混乱)
            for i in sorted(to_remove, reverse=True):
                del self.scheduled_changes[i]
    
    def clear_variable_override(self, variable_name):
        """
        清除变量覆盖
        
        参数:
            variable_name (str): 变量名
            
        返回:
            bool: 成功返回True，变量不存在返回False
        """
        with self.variable_lock:
            if variable_name in self.variable_overrides:
                del self.variable_overrides[variable_name]
                self.logger.info(f"已清除变量覆盖: {variable_name}")
                return True
            return False
    
    def simulate(self, duration):
        """
        执行固定时长的仿真
        
        参数:
            duration (float): 仿真时长(秒)
            
        返回:
            bool: 成功返回True，失败返回False
        """
        if not self.initialized:
            self.initialize()
        
        end_time = self.current_time + duration
        
        try:
            self.logger.info(f"开始仿真 {duration} 秒 (从 {self.current_time} 到 {end_time})")
            
            while self.current_time < end_time:
                success = self.step()
                if not success:
                    return False
            
            self.logger.info(f"仿真完成，共 {self.perf_metrics['total_steps']} 步")
            return True
        except Exception as e:
            self.logger.error(f"仿真失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_realtime(self):
        """
        开始实时仿真
        
        返回:
            bool: 成功返回True，失败返回False
        """
        if self.running:
            self.logger.warning("实时仿真已在运行中")
            return True
        
        if not self.initialized:
            success = self.initialize()
            if not success:
                return False
        
        self.running = True
        self.paused = False
        self.stop_flag.clear()
        
        self.rt_thread = threading.Thread(target=self._realtime_thread)
        self.rt_thread.daemon = True
        self.rt_thread.start()
        
        self.logger.info(f"实时仿真已启动，步长: {self.step_size}s，速度因子: {self.real_time_factor}")
        return True
    
    def _realtime_thread(self):
        """实时仿真线程内部方法"""
        self.logger.info("实时仿真线程已启动")
        last_sync_time = time.time()
        
        try:
            while self.running and not self.stop_flag.is_set():
                # 处理暂停状态
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # 执行一步仿真
                step_start = time.time()
                success = self.step()
                
                if not success:
                    self.logger.error("仿真步骤失败，停止实时仿真")
                    self.running = False
                    break
                
                # 计算要等待的时间
                elapsed = time.time() - step_start
                wait_time = (self.step_size / self.real_time_factor) - elapsed
                
                if wait_time > 0:
                    time.sleep(wait_time)
                else:
                    self.logger.debug(f"实时同步过载: {-wait_time:.3f}s")
                
                # 存储同步时间
                sync_time = time.time() - step_start
                self.perf_metrics['sync_time'].append(sync_time)
        
        except Exception as e:
            self.logger.error(f"实时仿真线程异常: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.running = False
            self.stop_flag.clear()
            self.logger.info("实时仿真线程已结束")
    
    def stop_realtime(self):
        """
        停止实时仿真
        
        返回:
            bool: 成功返回True，失败返回False
        """
        if not self.running:
            return True
        
        self.logger.info("正在停止实时仿真...")
        self.stop_flag.set()
        self.running = False
        
        # 等待线程结束
        if self.rt_thread and self.rt_thread.is_alive():
            self.rt_thread.join(timeout=5.0)
            if self.rt_thread.is_alive():
                self.logger.warning("实时仿真线程停止超时")
                return False
        
        self.logger.info("实时仿真已停止")
        return True
    
    def pause_realtime(self):
        """
        暂停实时仿真
        
        返回:
            bool: 成功返回True，失败返回False
        """
        if not self.running:
            self.logger.warning("实时仿真未运行")
            return False
        
        self.paused = True
        self.logger.info(f"实时仿真已暂停在 {self.current_time}s")
        return True
    
    def resume_realtime(self):
        """
        恢复实时仿真
        
        返回:
            bool: 成功返回True，失败返回False
        """
        if not self.running:
            self.logger.warning("实时仿真未运行")
            return False
        
        if not self.paused:
            return True
        
        self.paused = False
        self.logger.info(f"实时仿真已恢复，继续时间: {self.current_time}s")
        return True
    
    def restore_state(self, time_point=None):
        """
        恢复到指定时间点的状态
        
        参数:
            time_point (float): 要恢复到的时间点，默认为最近保存的状态
            
        返回:
            bool: 成功返回True，失败返回False
        """
        if not self.supports_state:
            self.logger.error("该模型不支持状态恢复")
            return False
        
        if not self.saved_states:
            self.logger.error("没有可用的保存状态")
            return False
        
        try:
            # 如果未指定时间点，使用最近保存的状态
            if time_point is None:
                time_point = max(self.saved_states.keys())
            
            # 如果指定时间点不存在，找最近的
            if time_point not in self.saved_states:
                # 找到小于等于指定时间点的最大值
                closest_points = [t for t in self.saved_states.keys() if t <= time_point]
                if not closest_points:
                    self.logger.error(f"找不到 {time_point}s 之前的保存状态")
                    return False
                
                time_point = max(closest_points)
                self.logger.info(f"使用最接近的状态: {time_point}s")
            
            # 恢复状态
            state = self.saved_states[time_point]
            self.model.set_fmu_state(state)
            
            # 更新当前时间
            self.current_time = time_point
            
            self.logger.info(f"已恢复到时间点: {time_point}s")
            return True
            
        except Exception as e:
            self.logger.error(f"恢复状态失败: {str(e)}")
            return False
    
    def save_current_state(self):
        """
        保存当前状态
        
        返回:
            bool: 成功返回True，失败返回False
        """
        if not self.supports_state:
            self.logger.error("该模型不支持状态保存")
            return False
        
        try:
            state = self.model.get_fmu_state()
            self.saved_states[self.current_time] = state
            self.logger.info(f"已保存时间点 {self.current_time}s 的状态")
            return True
        except Exception as e:
            self.logger.error(f"保存状态失败: {str(e)}")
            return False
    
    def get_variable_info(self, pattern=None, param_type=None):
        """
        获取模型变量信息
        
        参数:
            pattern (str): 可选的变量名过滤模式
            param_type (str): 可选的参数类型过滤 ('input', 'output', 'parameter')
            
        返回:
            list: 匹配的变量信息列表
        """
        all_vars = self.model.get_model_variables()
        result = []
        
        # 定义参数类型映射
        type_map = {
            'input': 2,
            'output': 3,
            'parameter': [0, 1]
        }
        
        for var_name in all_vars:
            match_pattern = pattern is None or re.search(pattern, var_name)
            
            if match_pattern:
                var_info = all_vars[var_name]
                info = {
                    'name': var_name,
                    'type': 'Unknown',
                    'causality': 'Unknown',
                    'variability': 'Unknown',
                    'description': 'None'
                }
                
                # 获取变量类型
                try:
                    info['type'] = self.model.get_variable_data_type(var_name)
                except:
                    pass
                
                # 获取变量因果性
                try:
                    causality = var_info.causality
                    causality_map = {
                        0: 'parameter',
                        1: 'calculated parameter',
                        2: 'input',
                        3: 'output',
                        4: 'local',
                        5: 'independent',
                        6: 'unknown'
                    }
                    info['causality'] = causality_map.get(causality, 'unknown')
                    
                    # 如果指定了参数类型过滤
                    if param_type is not None:
                        if param_type in type_map:
                            target_type = type_map[param_type]
                            if isinstance(target_type, list):
                                if causality not in target_type:
                                    continue
                            elif causality != target_type:
                                continue
                except:
                    if param_type is not None:
                        continue
                
                # 获取变量可变性
                try:
                    variability = var_info.variability
                    variability_map = {
                        0: 'constant',
                        1: 'fixed',
                        2: 'tunable',
                        3: 'discrete',
                        4: 'continuous',
                        5: 'unknown'
                    }
                    info['variability'] = variability_map.get(variability, 'unknown')
                except:
                    pass
                
                # 获取变量描述
                try:
                    info['description'] = self.model.get_variable_description(var_name)
                except:
                    pass
                
                # 获取变量值
                try:
                    info['value'] = self.model.get(var_name)
                except:
                    info['value'] = None
                
                # 是否可修改
                info['modifiable'] = var_name in self.modifiable_variables
                
                # 是否监控中
                info['monitored'] = var_name in self.variables
                
                result.append(info)
        
        return result
    
    def get_current_results(self):
        """
        获取当前仿真结果
        
        返回:
            dict: 包含当前时间和变量值的字典
        """
        result = {'time': self.current_time}
        
        for var in self.variables:
            try:
                result[var] = self.model.get(var)
            except Exception as e:
                self.logger.warning(f"获取变量 {var} 失败: {str(e)}")
                result[var] = None
        
        return result
    
    def get_result_history(self, variables=None, max_points=None):
        """
        获取仿真结果历史
        
        参数:
            variables (list): 要获取的变量列表，默认为所有监控变量
            max_points (int): 最大点数，默认返回所有
            
        返回:
            dict: 包含时间和变量历史值的字典
        """
        if variables is None:
            variables = self.variables
        
        # 获取时间序列
        times = list(self.results['time'])
        
        # 如果指定了最大点数，限制返回结果
        if max_points is not None and len(times) > max_points:
            # 选择均匀分布的点
            indices = np.linspace(0, len(times)-1, max_points).astype(int)
            times = [times[i] for i in indices]
        else:
            indices = range(len(times))
        
        # 构建结果
        result = {'time': times}
        
        for var in variables:
            if var in self.results:
                values = list(self.results[var])
                result[var] = [values[i] for i in indices]
        
        return result
    
    def plot_results(self, variables=None, figsize=(10, 6), grid=True, subplots=True, markers=True):
        """
        绘制仿真结果
    
        参数:
            variables (list): 要绘制的变量列表，默认为所有监控变量
            figsize (tuple): 图表大小
            grid (bool): 是否显示网格
            subplots (bool): 是否使用子图，默认为True
            markers (bool): 是否在数据点显示标记，默认为True
        """
        if not self.results['time']:
            self.logger.warning("No results available to plot")
            return None
        
        # 默认使用所有监控变量
        if variables is None:
            variables = self.variables
        
        # 获取结果
        times = list(self.results['time'])
        
        # 创建图表
        if subplots:
            fig, axes = plt.subplots(len(variables), 1, figsize=figsize, sharex=True)
            if len(variables) == 1:
                axes = [axes]  # 确保axes是列表
            
            for i, var in enumerate(variables):
                if var not in self.results:
                    self.logger.warning(f"Variable {var} not in results")
                    continue
                
                values = list(self.results[var])
                
                if markers:
                    axes[i].plot(times, values, '.-')
                else:
                    axes[i].plot(times, values)
                
                axes[i].set_ylabel(var)
                if grid:
                    axes[i].grid(True)
            
            axes[-1].set_xlabel('Time (s)')
            plt.tight_layout()
        else:
            fig = plt.figure(figsize=figsize)
            
            for var in variables:
                if var not in self.results:
                    self.logger.warning(f"Variable {var} not in results")
                    continue
                
                values = list(self.results[var])
                
                if markers:
                    plt.plot(times, values, '.-', label=var)
                else:
                    plt.plot(times, values, label=var)
            
            plt.xlabel('Time (s)')
            plt.ylabel('Value')
            plt.legend()
            if grid:
                plt.grid(True)
            plt.tight_layout()
        
        plt.suptitle(f"{self.model_name} Simulation Results")
        # 只显示一次图表，不在这里调用plt.show()
        # 让调用者决定何时显示图表
        plt.show()
        
        return fig
    
    def export_results(self, filename, format='csv'):
        """
        导出仿真结果
        
        参数:
            filename (str): 输出文件名
            format (str): 输出格式，支持'csv'和'excel'
            
        返回:
            bool: 成功返回True，失败返回False
        """
        try:
            # 获取结果
            times = list(self.results['time'])
            
            if not times:
                self.logger.warning("没有可导出的结果")
                return False
            
            # 构建数据表
            data = {'time': times}
            for var in self.variables:
                if var in self.results:
                    data[var] = list(self.results[var])
            
            if format.lower() == 'csv':
                # 导出为CSV
                import csv
                
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # 写入表头
                    writer.writerow(['time'] + self.variables)
                    # 写入数据
                    for i in range(len(times)):
                        row = [times[i]]
                        for var in self.variables:
                            if var in self.results and i < len(self.results[var]):
                                row.append(self.results[var][i])
                            else:
                                row.append('')
                        writer.writerow(row)
                
                self.logger.info(f"结果已导出到CSV文件: {filename}")
                return True
                
            elif format.lower() == 'excel':
                # 导出为Excel
                try:
                    import pandas as pd
                    
                    # 构建DataFrame
                    df = pd.DataFrame(data)
                    
                    # 导出到Excel
                    df.to_excel(filename, index=False)
                    
                    self.logger.info(f"结果已导出到Excel文件: {filename}")
                    return True
                    
                except ImportError:
                    self.logger.error("导出到Excel需要安装pandas库")
                    return False
            else:
                self.logger.error(f"不支持的格式: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"导出结果失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def close(self):
        """关闭模拟器，释放资源"""
        self.logger.info("正在关闭模拟器...")
        
        # 关闭WebSocket服务器
        if hasattr(self, 'ws_server') and self.ws_server is not None:
            self.logger.info("正在关闭WebSocket服务器...")
            self.ws_server.shutdown()
            self.ws_server.server_close()
            self.logger.info("WebSocket服务器已关闭")
        
        # 关闭模型 - 检查是否有terminate方法而不是close
        if hasattr(self, 'model') and self.model is not None:
            # 对于PyFMI模型，应该调用terminate而不是close
            if hasattr(self.model, 'terminate'):
                self.model.terminate()
            # 如果模型有free_instance方法，也可以调用
            if hasattr(self.model, 'free_instance'):
                self.model.free_instance()
            # 如果还有其他PyFMI模型清理方法，可以在此添加
        
        # 关闭线程池
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        
        # 清理剩余资源
        self.logger.info("模拟器已关闭")
    
    # ===== v1.9版本新增功能：WebSocket接口 =====
    
    def _ensure_json_serializable(self, data):
        """确保数据可以序列化为JSON (内部方法)"""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = self._ensure_json_serializable(value)
            return result
        elif isinstance(data, list):
            return [self._ensure_json_serializable(item) for item in data]
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return self._ensure_json_serializable(data.tolist())
        elif isinstance(data, deque):
            return self._ensure_json_serializable(list(data))
        else:
            # 对于其他类型，尝试转换为字符串
            return str(data)
    
    async def _websocket_handler(self, websocket):
        """WebSocket连接处理函数 (内部方法) - 适用于websockets 15.0.1"""
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.logger.info(f"WebSocket客户端已连接: {client_info}")
        
        try:
            # 添加客户端到列表
            with self.ws_lock:
                self.ws_clients.add(websocket)
            
            # 发送初始状态
            try:
                initial_info = {
                    'type': 'info',
                    'model_name': self.model_name,
                    'variables': self.variables,
                    'current_time': float(self.current_time),
                    'running': bool(self.running),
                    'paused': bool(self.paused)
                }
                
                # 确保所有值都可以序列化为JSON
                initial_info = self._ensure_json_serializable(initial_info)
                
                self.logger.debug(f"发送初始信息: {initial_info}")
                await websocket.send(json.dumps(initial_info))
                self.logger.debug("初始信息已发送")
            except Exception as e:
                self.logger.error(f"发送初始信息失败: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # 处理客户端消息
            async for message in websocket:
                self.logger.debug(f"收到消息: {message}")
                
                try:
                    # 解析JSON消息
                    data = json.loads(message)
                    command = data.get('command')
                    
                    self.logger.info(f"收到命令: {command}")
                    
                    # 处理不同命令
                    if command == 'start':
                        success = self.start_realtime()
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'command': 'start',
                            'success': success
                        }))
                    
                    elif command == 'stop':
                        success = self.stop_realtime()
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'command': 'stop',
                            'success': success
                        }))
                    
                    elif command == 'pause':
                        success = self.pause_realtime()
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'command': 'pause',
                            'success': success
                        }))
                    
                    elif command == 'resume':
                        success = self.resume_realtime()
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'command': 'resume',
                            'success': success
                        }))
                    
                    elif command == 'step':
                        success = self.step()
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'command': 'step',
                            'success': success
                        }))
                    
                    elif command == 'set_variable':
                        var_name = data.get('variable')
                        value = data.get('value')
                        
                        if var_name and value is not None:
                            success = self.set_variable(var_name, value)
                            await websocket.send(json.dumps({
                                'type': 'response',
                                'command': 'set_variable',
                                'variable': var_name,
                                'success': success
                            }))
                    
                    elif command == 'get_results':
                        # 获取最新结果
                        results = self.get_current_results()
                        # 确保数据可序列化
                        results = self._ensure_json_serializable(results)
                        await websocket.send(json.dumps({
                            'type': 'results',
                            'data': results
                        }))
                    
                    elif command == 'get_history':
                        # 获取历史结果
                        variables = data.get('variables')
                        max_points = data.get('max_points')
                        
                        history = self.get_result_history(variables, max_points)
                        # 确保数据可序列化
                        history = self._ensure_json_serializable(history)
                        await websocket.send(json.dumps({
                            'type': 'history',
                            'data': history
                        }))
    
                    elif command == 'get_variable_info':
                        pattern = data.get('pattern')
                        param_type = data.get('param_type')  # 新增参数类型过滤
                        info = self.get_variable_info(pattern, param_type)
                        # 确保数据可序列化
                        info = self._ensure_json_serializable(info)
                        await websocket.send(json.dumps({
                            'type': 'variable_info',
                            'data': info
                        }))
                    
                    elif command == 'get_status':
                        status = {
                            'running': self.running,
                            'paused': self.paused,
                            'current_time': self.current_time,
                            'step_size': self.step_size,
                            'real_time_factor': self.real_time_factor,
                            'perf_metrics': {
                                'total_steps': self.perf_metrics['total_steps'],
                                'overruns': self.perf_metrics['overruns'],
                                'avg_step_time': np.mean(self.perf_metrics['step_time']) if self.perf_metrics['step_time'] else 0,
                                'avg_sync_time': np.mean(self.perf_metrics['sync_time']) if self.perf_metrics['sync_time'] else 0
                            }
                        }
                        # 确保数据可序列化
                        status = self._ensure_json_serializable(status)
                        await websocket.send(json.dumps({
                            'type': 'status',
                            'data': status
                        }))
                    
                    elif command == 'set_real_time_factor':
                        factor = data.get('factor')
                        if factor is not None:
                            self.real_time_factor = float(factor)
                            await websocket.send(json.dumps({
                                'type': 'response',
                                'command': 'set_real_time_factor',
                                'success': True
                            }))
                    
                    else:
                        self.logger.warning(f"未知命令: {command}")
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': f"未知命令: {command}"
                        }))
                
                except json.JSONDecodeError:
                    self.logger.error("无效的JSON格式")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': "无效的JSON格式"
                    }))
                
                except Exception as e:
                    self.logger.error(f"处理命令失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': f"处理命令失败: {str(e)}"
                    }))
        
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.info(f"WebSocket连接已关闭: {client_info} - {e}")
        
        except Exception as e:
            self.logger.error(f"WebSocket连接异常: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 移除客户端
            with self.ws_lock:
                if websocket in self.ws_clients:
                    self.ws_clients.remove(websocket)
            self.logger.info(f"WebSocket客户端已断开: {client_info}")
    
    def _broadcast_results(self):
        """广播仿真结果到WebSocket客户端 (内部方法)"""
        if not self.ws_clients:
            return
        
        try:
            # 准备要发送的数据
            results = self.get_current_results()
            
            # 确保数据可序列化
            results = self._ensure_json_serializable(results)
            
            message = {
                'type': 'results',
                'data': results
            }
            
            # 序列化为JSON
            json_msg = json.dumps(message)
            
            # 提交任务到线程池，异步发送给所有客户端
            self.executor.submit(self._send_to_all_clients, json_msg)
            
        except Exception as e:
            self.logger.error(f"广播结果失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _send_to_all_clients(self, message):
        """发送消息到所有WebSocket客户端 (内部方法)"""
        if not hasattr(self, 'ws_loop') or self.ws_loop is None:
            self.logger.error("WebSocket事件循环未初始化")
            return
                
        with self.ws_lock:
            for websocket in list(self.ws_clients):
                try:
                    # 创建异步任务运行在事件循环中
                    asyncio.run_coroutine_threadsafe(
                        websocket.send(message),
                        self.ws_loop
                    )
                except Exception as e:
                    self.logger.error(f"发送消息失败: {str(e)}")
    
    async def _start_websocket_server(self, host, port):
        """启动WebSocket服务器 (内部方法)"""
        try:
            self.logger.info(f"正在启动WebSocket服务器: {host}:{port}")
            
            # 添加origins参数允许所有来源
            server = await websockets.serve(
                self._websocket_handler,
                host,
                port,
                origins=None  # 允许所有来源
            )
            
            self.ws_server = server
            self.ws_running = True
            self.logger.info(f"WebSocket服务器已启动: ws://{host}:{port}")
            
            # 保持服务器运行
            while self.ws_running:
                await asyncio.sleep(1)
            
            # 关闭服务器
            self.logger.info("正在关闭WebSocket服务器...")
            server.close()
            await server.wait_closed()
            self.logger.info("WebSocket服务器已关闭")
            
        except Exception as e:
            self.logger.error(f"启动WebSocket服务器失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.ws_running = False

        # 在_start_websocket_server方法中
        server = await websockets.serve(
            self._websocket_handler,
            host,
            port,
            ping_interval=30,  # 30秒发送一次ping
            ping_timeout=10    # 10秒内没有pong则断开
        )
    
    def _run_websocket_loop(self, host, port):
        """在线程中运行WebSocket服务器 (内部方法)"""
        # 设置线程的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.ws_loop = loop
        
        try:
            # 运行服务器
            loop.run_until_complete(self._start_websocket_server(host, port))
        except Exception as e:
            self.logger.error(f"WebSocket服务器线程异常: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            loop.close()
            self.logger.info("WebSocket服务器线程已结束")
    
    def start_websocket_server(self, host='localhost', port=8765):
        """
        启动WebSocket服务器
        
        参数:
            host (str): 服务器主机名，默认为localhost
            port (int): 服务器端口，默认为8765
            
        返回:
            bool: 成功返回True，失败返回False
        """
        if self.ws_running:
            self.logger.warning("WebSocket服务器已在运行")
            return True
        
        try:
            # 确保已导入nest_asyncio并应用
            try:
                import nest_asyncio
                nest_asyncio.apply()
                self.logger.info("已应用nest_asyncio")
            except ImportError:
                self.logger.warning("无法导入nest_asyncio，在Jupyter中可能会出现事件循环问题")
            
            self.logger.info(f"正在启动WebSocket服务器线程: {host}:{port}")
            # 创建并启动WebSocket服务器线程
            self.ws_thread = threading.Thread(
                target=self._run_websocket_loop,
                args=(host, port),
                daemon=True
            )
            self.ws_thread.start()
            
            # 等待服务器启动
            timeout = 5
            start_time = time.time()
            while not self.ws_running and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if not self.ws_running:
                self.logger.error("WebSocket服务器启动超时")
                return False
            
            self.logger.info(f"WebSocket服务器已成功启动: ws://{host}:{port}")
            return True
                
        except Exception as e:
            self.logger.error(f"启动WebSocket服务器失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop_websocket_server(self):
        """
        停止WebSocket服务器
        
        返回:
            bool: 成功返回True，失败返回False
        """
        if not self.ws_running:
            return True
        
        try:
            # 设置停止标志
            self.logger.info("正在停止WebSocket服务器...")
            self.ws_running = False
            
            # 等待线程结束
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5.0)
                if self.ws_thread.is_alive():
                    self.logger.warning("WebSocket服务器停止超时")
            
            self.logger.info("WebSocket服务器已停止")
            return True
                
        except Exception as e:
            self.logger.error(f"停止WebSocket服务器失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    ## ===== v1.9版本新增功能：http 接口 =====

    def start_http_server(self, host='localhost', port=5000):
        """
        启动HTTP服务器
        
        参数:
            host (str): 服务器主机名，默认为localhost
            port (int): 服务器端口，默认为5000
            
        返回:
            bool: 成功返回True，失败返回False
        """
        try:
            # 导入HTTP服务器类
            import sys
            sys.path.append('fmu_http_server_Pro.py')  # 替换为你的模块路径
            from fmu_http_server_Pro import FmuHttpServer
            
            # 创建并启动HTTP服务器
            self.http_server = FmuHttpServer(self)
            success = self.http_server.start(host, port)
            
            if success:
                self.logger.info(f"HTTP服务器已启动: http://{host}:{port}")
            
            return success
        except Exception as e:
            self.logger.error(f"启动HTTP服务器失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop_http_server(self):
        """
        停止HTTP服务器
        
        返回:
            bool: 成功返回True，失败返回False
        """
        try:
            if hasattr(self, 'http_server') and self.http_server:
                success = self.http_server.stop()
                if success:
                    self.logger.info("HTTP服务器已停止")
                return success
            return True
        except Exception as e:
            self.logger.error(f"停止HTTP服务器失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    
    # ===== v1.9版本新增功能：模型诊断与辅助 =====
    
    def diagnose_fmu(self):
        """
        诊断FMU模型
        
        返回:
            dict: 诊断结果
        """
        diagnosis = {
            'model_name': self.model_name,
            'supports_state': self.supports_state,
            'variables_count': self._count_variables_by_type(),
            'performance': self._get_performance_stats(),
            'issues': []
        }
        
        # 检查变量
        if not self.variables:
            diagnosis['issues'].append("未监控任何变量")
        
        # 检查可修改变量
        if not self.modifiable_variables:
            diagnosis['issues'].append("未找到可修改的变量")
        
        # 检查性能
        if self.perf_metrics['overruns'] > 0:
            overrun_pct = (self.perf_metrics['overruns'] / max(1, self.perf_metrics['total_steps'])) * 100
            if overrun_pct > 10:
                diagnosis['issues'].append(f"性能过载: {overrun_pct:.1f}% 的步骤无法实时完成")
        
        # 尝试获取模型信息
        try:
            diagnosis['fmi_version'] = self.model.get_version()
        except:
            diagnosis['fmi_version'] = "Unknown"
            diagnosis['issues'].append("无法获取FMI版本")
        
        # 检查变量是否可读取
        unreadable = []
        for var in self.variables:
            try:
                self.model.get(var)
            except:
                unreadable.append(var)
        
        if unreadable:
            diagnosis['issues'].append(f"以下变量无法读取: {', '.join(unreadable)}")
        
        return diagnosis
    
    def _count_variables_by_type(self):
        """统计各类型变量数量 (内部方法)"""
        all_vars = self.model.get_model_variables()
        counts = {
            'total': len(all_vars),
            'inputs': 0,
            'outputs': 0,
            'parameters': 0,
            'others': 0,
            'monitored': len(self.variables),
            'modifiable': len(self.modifiable_variables)
        }
        
        for var in all_vars:
            try:
                causality = all_vars[var].causality
                if causality == 2:  # input
                    counts['inputs'] += 1
                elif causality == 3:  # output
                    counts['outputs'] += 1
                elif causality == 0 or causality == 1:  # parameter
                    counts['parameters'] += 1
                else:
                    counts['others'] += 1
            except:
                counts['others'] += 1
        
        return counts
    
    def _get_performance_stats(self):
        """获取性能统计 (内部方法)"""
        stats = {
            'total_steps': self.perf_metrics['total_steps'],
            'overruns': self.perf_metrics['overruns']
        }
        
        if self.perf_metrics['step_time']:
            step_times = np.array(self.perf_metrics['step_time'])
            stats['avg_step_time'] = np.mean(step_times)
            stats['max_step_time'] = np.max(step_times)
            stats['min_step_time'] = np.min(step_times)
        
        if self.perf_metrics['sync_time']:
            sync_times = np.array(self.perf_metrics['sync_time'])
            stats['avg_sync_time'] = np.mean(sync_times)
            stats['max_sync_time'] = np.max(sync_times)
        
        return stats
    
    def suggest_monitoring_variables(self, max_count=10):
        """
        建议监控变量
        
        参数:
            max_count (int): 最大建议变量数量
            
        返回:
            list: 建议监控的变量列表
        """
        all_vars = self.model.get_model_variables()
        suggestions = []
        
        # 优先级顺序: 输出 > 状态 > 重要参数
        # 1. 首先添加输出变量
        for var in all_vars:
            try:
                if len(suggestions) >= max_count:
                    break
                
                var_info = all_vars[var]
                if var_info.causality == 3:  # output
                    if var not in suggestions:
                        suggestions.append(var)
            except:
                pass
        
        # 2. 添加可能是状态的变量
        for var in all_vars:
            if len(suggestions) >= max_count:
                break
            
            if var not in suggestions and ('state' in var.lower() or 'x[' in var.lower()):
                suggestions.append(var)
        
        # 3. 添加重要参数或看起来重要的变量
        important_keywords = ['result', 'out', 'final', 'value', 'temp', 'time', 'position', 'speed']
        for keyword in important_keywords:
            if len(suggestions) >= max_count:
                break
            
            for var in all_vars:
                if len(suggestions) >= max_count:
                    break
                
                if var not in suggestions and keyword in var.lower():
                    suggestions.append(var)
        
        return suggestions
    
    def set_step_size(self, step_size):
        """
        设置仿真步长
        
        参数:
            step_size (float): 新的步长值
            
        返回:
            bool: 成功返回True，失败返回False
        """
        if step_size <= 0:
            self.logger.error("步长必须大于0")
            return False
        
        self.step_size = step_size
        self.logger.info(f"步长已设置为: {step_size}s")
        return True
    
    def add_monitor_variable(self, variable_name):
        """
        添加监控变量
        
        参数:
            variable_name (str): 变量名
            
        返回:
            bool: 成功返回True，失败返回False
        """
        try:
            # 检查变量是否存在
            self.model.get(variable_name)
            
            # 添加到监控列表
            if variable_name not in self.variables:
                self.variables.append(variable_name)
                self.results[variable_name] = deque(maxlen=10000)
                
                # 填充历史数据为None
                for _ in range(len(self.results['time'])):
                    self.results[variable_name].append(None)
                
                self.logger.info(f"已添加监控变量: {variable_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"添加监控变量失败: {str(e)}")
            return False
    
    def remove_monitor_variable(self, variable_name):
        """
        移除监控变量
        
        参数:
            variable_name (str): 变量名
            
        返回:
            bool: 成功返回True，失败返回False
        """
        if variable_name in self.variables:
            self.variables.remove(variable_name)
            if variable_name in self.results:
                del self.results[variable_name]
            
            self.logger.info(f"已移除监控变量: {variable_name}")
            return True
        
        return False