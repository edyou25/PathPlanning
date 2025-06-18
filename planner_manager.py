import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum
import time

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class PlannerType(Enum):
    """支持的规划器类型"""
    # 基于搜索的2D规划器
    A_STAR_2D = "a_star_2d"
    BIDIRECTIONAL_A_STAR_2D = "bidirectional_a_star_2d"
    ARA_STAR_2D = "ara_star_2d"
    
    # 基于搜索的3D规划器
    A_STAR_3D = "a_star_3d"
    BIDIRECTIONAL_A_STAR_3D = "bidirectional_a_star_3d"
    D_STAR_LITE_3D = "d_star_lite_3d"
    ANYTIME_D_STAR_3D = "anytime_d_star_3d"
    LRT_A_STAR_3D = "lrt_a_star_3d"
    RTA_A_STAR_3D = "rta_a_star_3d"
    LP_A_STAR_3D = "lp_a_star_3d"
    
    # 基于采样的2D规划器
    RRT_2D = "rrt_2d"
    RRT_STAR_2D = "rrt_star_2d"
    RRT_STAR_SMART_2D = "rrt_star_smart_2d"
    DUBINS_RRT_STAR_2D = "dubins_rrt_star_2d"
    FMT_2D = "fmt_2d"
    
    # 基于采样的3D规划器
    RRT_3D = "rrt_3d"
    RRT_CONNECT_3D = "rrt_connect_3d"
    RRT_STAR_3D = "rrt_star_3d"
    ABIT_STAR_3D = "abit_star_3d"
    FMT_STAR_3D = "fmt_star_3d"
    
    # 曲线生成器
    BEZIER_PATH = "bezier_path"
    BSPLINE_CURVE = "bspline_curve"
    CUBIC_SPLINE = "cubic_spline"
    DUBINS_PATH = "dubins_path"
    REEDS_SHEPP = "reeds_shepp"
    QUARTIC_POLYNOMIAL = "quartic_polynomial"
    QUINTIC_POLYNOMIAL = "quintic_polynomial"

class PlannerDimension(Enum):
    """规划器维度"""
    TWO_D = "2D"
    THREE_D = "3D"
    CURVE = "CURVE"

class PlannerCategory(Enum):
    """规划器分类"""
    SEARCH_BASED = "search_based"
    SAMPLING_BASED = "sampling_based"
    CURVE_GENERATOR = "curve_generator"

class PathPlannerManager:
    """路径规划管理器 - 统一调用接口"""
    
    def __init__(self):
        self.current_planner = None
        self.current_planner_type = None
        self.planning_results = {}
        self.planner_configs = self._load_default_configs()
        
    def _load_default_configs(self) -> Dict[PlannerType, Dict[str, Any]]:
        """加载默认配置"""
        configs = {}
        
        # 2D搜索算法配置
        configs[PlannerType.A_STAR_2D] = {
            'heuristic_type': 'manhattan',
            'resolution': 1.0
        }
        
        configs[PlannerType.BIDIRECTIONAL_A_STAR_2D] = {
            'heuristic_type': 'manhattan',
            'resolution': 1.0
        }
        
        # 3D搜索算法配置
        configs[PlannerType.A_STAR_3D] = {
            'resolution': 0.5,
            'weight': 1.0
        }
        
        configs[PlannerType.BIDIRECTIONAL_A_STAR_3D] = {
            'resolution': 0.5,
            'weight': 1.0
        }
        
        # 2D采样算法配置
        configs[PlannerType.RRT_STAR_2D] = {
            'step_len': 0.5,
            'goal_sample_rate': 0.1,
            'search_radius': 1.0,
            'iter_max': 2000
        }
        
        configs[PlannerType.DUBINS_RRT_STAR_2D] = {
            'step_len': 30.0,
            'goal_sample_rate': 0.1,
            'search_radius': 50.0,
            'iter_max': 250,
            'vehicle_radius': 2.0
        }
        
        configs[PlannerType.RRT_STAR_SMART_2D] = {
            'step_len': 0.5,
            'goal_sample_rate': 0.1,
            'search_radius': 1.0,
            'iter_max': 2000
        }
        
        # 3D采样算法配置
        configs[PlannerType.RRT_CONNECT_3D] = {
            'max_iter': 1000,
            'step_size': 0.5,
            'goal_bias': 0.1
        }
        
        configs[PlannerType.ABIT_STAR_3D] = {
            'max_iter': 1000,
            'step_size': 0.5,
            'goal_bias': 0.1
        }
        
        # 曲线生成器配置
        configs[PlannerType.CUBIC_SPLINE] = {
            'resolution': 0.1
        }
        
        configs[PlannerType.BEZIER_PATH] = {
            'offset': 3.0
        }
        
        configs[PlannerType.DUBINS_PATH] = {
            'curvature': 1.0,
            'step_size': 0.1
        }
        
        return configs
    
    def set_planner_config(self, planner_type: PlannerType, config: Dict[str, Any]):
        """设置规划器配置"""
        self.planner_configs[planner_type] = config
    
    def get_planner_info(self, planner_type: PlannerType) -> Dict[str, str]:
        """获取规划器信息"""
        info_map = {
            # 2D搜索算法
            PlannerType.A_STAR_2D: {
                'name': 'A* 2D',
                'dimension': PlannerDimension.TWO_D.value,
                'category': PlannerCategory.SEARCH_BASED.value,
                'description': '2D A*搜索算法'
            },
            PlannerType.BIDIRECTIONAL_A_STAR_2D: {
                'name': 'Bidirectional A* 2D',
                'dimension': PlannerDimension.TWO_D.value,
                'category': PlannerCategory.SEARCH_BASED.value,
                'description': '2D双向A*搜索算法'
            },
            
            # 3D搜索算法
            PlannerType.A_STAR_3D: {
                'name': 'A* 3D',
                'dimension': PlannerDimension.THREE_D.value,
                'category': PlannerCategory.SEARCH_BASED.value,
                'description': '3D A*搜索算法'
            },
            PlannerType.BIDIRECTIONAL_A_STAR_3D: {
                'name': 'Bidirectional A* 3D',
                'dimension': PlannerDimension.THREE_D.value,
                'category': PlannerCategory.SEARCH_BASED.value,
                'description': '3D双向A*搜索算法'
            },
            
            # 2D采样算法
            PlannerType.RRT_STAR_2D: {
                'name': 'RRT* 2D',
                'dimension': PlannerDimension.TWO_D.value,
                'category': PlannerCategory.SAMPLING_BASED.value,
                'description': '2D RRT*采样算法'
            },
            PlannerType.DUBINS_RRT_STAR_2D: {
                'name': 'Dubins RRT* 2D',
                'dimension': PlannerDimension.TWO_D.value,
                'category': PlannerCategory.SAMPLING_BASED.value,
                'description': '2D Dubins RRT*采样算法（考虑车辆转弯约束）'
            },
            PlannerType.RRT_STAR_SMART_2D: {
                'name': 'RRT* Smart 2D',
                'dimension': PlannerDimension.TWO_D.value,
                'category': PlannerCategory.SAMPLING_BASED.value,
                'description': '2D RRT* Smart采样算法（智能采样）'
            },
            
            # 3D采样算法
            PlannerType.RRT_CONNECT_3D: {
                'name': 'RRT-Connect 3D',
                'dimension': PlannerDimension.THREE_D.value,
                'category': PlannerCategory.SAMPLING_BASED.value,
                'description': '3D RRT-Connect双向采样算法'
            },
            PlannerType.ABIT_STAR_3D: {
                'name': 'ABIT* 3D',
                'dimension': PlannerDimension.THREE_D.value,
                'category': PlannerCategory.SAMPLING_BASED.value,
                'description': '3D ABIT*采样算法（先进批量信息树）'
            },
            
            # 曲线生成器
            PlannerType.CUBIC_SPLINE: {
                'name': 'Cubic Spline',
                'dimension': PlannerDimension.CURVE.value,
                'category': PlannerCategory.CURVE_GENERATOR.value,
                'description': '三次样条曲线生成器'
            },
            PlannerType.BEZIER_PATH: {
                'name': 'Bezier Path',
                'dimension': PlannerDimension.CURVE.value,
                'category': PlannerCategory.CURVE_GENERATOR.value,
                'description': '贝塞尔曲线路径生成器'
            },
            PlannerType.DUBINS_PATH: {
                'name': 'Dubins Path',
                'dimension': PlannerDimension.CURVE.value,
                'category': PlannerCategory.CURVE_GENERATOR.value,
                'description': 'Dubins路径生成器（最短转弯路径）'
            }
        }
        return info_map.get(planner_type, {'name': 'Unknown', 'dimension': 'Unknown', 'category': 'Unknown', 'description': 'Unknown planner'})
    
    def plan_path(self, 
                  start: Tuple[float, ...], 
                  goal: Tuple[float, ...], 
                  planner_type: PlannerType,
                  config_override: Optional[Dict] = None) -> Optional[List[Tuple[float, ...]]]:
        """
        统一的路径规划接口
        
        Args:
            start: 起始点坐标
            goal: 目标点坐标
            planner_type: 规划器类型
            config_override: 配置覆盖参数
            
        Returns:
            规划的路径点列表或None
        """
        # 合并配置
        config = self.planner_configs.get(planner_type, {}).copy()
        if config_override:
            config.update(config_override)
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 根据规划器类型调用对应的规划算法
            path = self._call_planner(start, goal, planner_type, config)
            
            # 记录结果
            planning_time = time.time() - start_time
            self.planning_results[planner_type] = {
                'path': path,
                'planning_time': planning_time,
                'success': path is not None,
                'path_length': self._calculate_path_length(path) if path else 0
            }
            
            print(f"规划器 {planner_type.value} 完成，用时: {planning_time:.3f}s")
            if path:
                print(f"路径长度: {len(path)} 点，总距离: {self.planning_results[planner_type]['path_length']:.3f}")
            else:
                print("路径规划失败")
            
            return path
            
        except Exception as e:
            import traceback
            print(f"规划器 {planner_type.value} 执行失败:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print("详细错误堆栈:")
            print(traceback.format_exc())
            
            # 记录失败结果
            planning_time = time.time() - start_time
            self.planning_results[planner_type] = {
                'path': None,
                'planning_time': planning_time,
                'success': False,
                'path_length': 0,
                'error': str(e),
                'error_type': type(e).__name__
            }
            return None
    
    def _call_planner(self, start, goal, planner_type, config):
        """调用具体的规划器"""
        # 2D搜索算法
        if planner_type == PlannerType.A_STAR_2D:
            return self._call_astar_2d(start, goal, config)
        elif planner_type == PlannerType.BIDIRECTIONAL_A_STAR_2D:
            return self._call_bidirectional_astar_2d(start, goal, config)
        
        # 3D搜索算法
        elif planner_type == PlannerType.A_STAR_3D:
            return self._call_astar_3d(start, goal, config)
        elif planner_type == PlannerType.BIDIRECTIONAL_A_STAR_3D:
            return self._call_bidirectional_astar_3d(start, goal, config)
        
        # 2D采样算法
        elif planner_type == PlannerType.RRT_STAR_2D:
            return self._call_rrt_star_2d(start, goal, config)
        elif planner_type == PlannerType.DUBINS_RRT_STAR_2D:
            return self._call_dubins_rrt_star_2d(start, goal, config)
        elif planner_type == PlannerType.RRT_STAR_SMART_2D:
            return self._call_rrt_star_smart_2d(start, goal, config)
        
        # 3D采样算法
        elif planner_type == PlannerType.RRT_CONNECT_3D:
            return self._call_rrt_connect_3d(start, goal, config)
        elif planner_type == PlannerType.ABIT_STAR_3D:
            return self._call_abit_star_3d(start, goal, config)
        
        # 曲线生成器
        elif planner_type == PlannerType.CUBIC_SPLINE:
            return self._call_cubic_spline(start, goal, config)
        elif planner_type == PlannerType.BEZIER_PATH:
            return self._call_bezier_path(start, goal, config)
        elif planner_type == PlannerType.DUBINS_PATH:
            return self._call_dubins_path(start, goal, config)
        else:
            raise NotImplementedError(f"规划器 {planner_type.value} 尚未实现")
    
    def _call_astar_2d(self, start, goal, config):
        """调用2D A*算法"""
        # 将浮点坐标转换为整数坐标（A*算法期望整数格网坐标）
        start_int = (int(round(start[0])), int(round(start[1])))
        goal_int = (int(round(goal[0])), int(round(goal[1])))
        
        print(f"原始坐标 - 起始点: {start}, 目标点: {goal}")
        print(f"转换坐标 - 起始点: {start_int}, 目标点: {goal_int}")
        
        # 导入A*算法
        from Search_based_Planning.Search_2D.Astar import AStar
        
        # 尝试不同的构造函数参数组合
        try:
            planner = AStar(start_int, goal_int, config.get('heuristic_type', 'manhattan'))
        except TypeError:
            # 如果参数不匹配，尝试其他方式
            try:
                planner = AStar(s_start=start_int, s_goal=goal_int, heuristic_type=config.get('heuristic_type', 'manhattan'))
            except TypeError:
                # 如果还是不行，尝试最简单的参数
                planner = AStar(start_int, goal_int)
        
        path, _ = planner.searching()
        
        # 将整数路径坐标转换回浮点数坐标
        if path:
            path = [(float(p[0]), float(p[1])) for p in path]
        
        return path
    
    def _call_astar_3d(self, start, goal, config):
        """调用3D A*算法"""
        # 添加必要的路径到sys.path
        search_3d_path = os.path.join(os.path.dirname(__file__), 'Search_based_Planning', 'Search_3D')
        if search_3d_path not in sys.path:
            sys.path.append(search_3d_path)
        
        # 确保搜索路径正确
        search_base_path = os.path.join(os.path.dirname(__file__), 'Search_based_Planning')
        if search_base_path not in sys.path:
            sys.path.append(search_base_path)
        
        # 导入3D A*算法
        from Search_based_Planning.Search_3D.Astar3D import Weighted_A_star
        
        planner = Weighted_A_star(resolution=config.get('resolution', 0.5))
        
        # 设置起始点和目标点
        planner.env.start = np.array(start)
        planner.env.goal = np.array(goal)
        planner.start, planner.goal = tuple(start), tuple(goal)
        planner.x0, planner.xt = planner.start, planner.goal
        
        success = planner.run()
        if success and planner.Path:
            # 3D A*的Path格式是[[current_point, parent_point], ...]的边列表
            # 需要提取出连续的路径点
            path_points = []
            
            # 从路径边重建完整路径
            if planner.Path:
                # 添加最后一个点（目标点附近）
                path_points.append(planner.lastpoint)
                
                # 逆向遍历路径，从目标到起始
                current = planner.lastpoint
                while current in planner.Parent:
                    parent = planner.Parent[current]
                    path_points.append(parent)
                    current = parent
                
                # 反转路径，使其从起始到目标
                path_points.reverse()
                
                # 转换为浮点坐标
                return [(float(p[0]), float(p[1]), float(p[2])) for p in path_points]
        return None
    
    def _call_rrt_star_2d(self, start, goal, config):
        """调用2D RRT*算法"""
        from Sampling_based_Planning.rrt_2D.rrt_star import RrtStar
        
        planner = RrtStar(
            x_start=start,
            x_goal=goal,
            step_len=config.get('step_len', 10),
            goal_sample_rate=config.get('goal_sample_rate', 0.1),
            search_radius=config.get('search_radius', 1.0),
            iter_max=config.get('iter_max', 500)
        )
        
        path = planner.planning()
        return path
    
    def _call_rrt_connect_3d(self, start, goal, config):
        """调用3D RRT-Connect算法"""
        from Sampling_based_Planning.rrt_3D.rrt_connect3D import rrt_connect
        
        planner = rrt_connect()
        planner.maxiter = config.get('max_iter', 1000)
        
        # Set start and goal in environment
        planner.env.start = np.array(start)
        planner.env.goal = np.array(goal)
        
        result = planner.RRT_CONNECT_PLANNER(start, goal)
        if planner.done and planner.Path:
            # Convert to float tuples
            path = [(float(p[0]), float(p[1]), float(p[2])) for p in planner.Path]
            return path
        return None
    
    def _call_cubic_spline(self, start, goal, config):
        """调用三次样条曲线生成器"""
        from CurvesGenerator.cubic_spline import Spline2D
        
        # 创建简单的waypoints
        waypoints = [start, goal]
        waypoints = self._call_astar_2d(start, goal, config)
        
        # 如果只有两个点，添加中间点
        if len(waypoints) == 2:
            mid_x = (start[0] + goal[0]) / 2
            mid_y = (start[1] + goal[1]) / 2
            waypoints = [start, (mid_x, mid_y), goal]
        
        x_coords = [p[0] for p in waypoints]
        y_coords = [p[1] for p in waypoints]
        
        spline = Spline2D(x_coords, y_coords)
        
        # 生成路径点
        resolution = config.get('resolution', 0.1)
        t_values = np.arange(0, spline.s[-1], resolution)
        path = []
        
        for t in t_values:
            if t < len(t_values) - 1:
                x, y = spline.calc_position(t)
                path.append((float(x), float(y)))
        
        return path
    
    def _call_dubins_rrt_star_2d(self, start, goal, config):
        """调用2D Dubins RRT*算法"""
        from Sampling_based_Planning.rrt_2D.dubins_rrt_star import DubinsRRTStar
        import numpy as np
        
        # 默认偏航角
        sx, sy, syaw = start[0], start[1], np.deg2rad(90)
        gx, gy, gyaw = goal[0], goal[1], np.deg2rad(0)
        
        # 如果起始点和目标点包含偏航角信息
        if len(start) >= 3:
            syaw = start[2]
        if len(goal) >= 3:
            gyaw = goal[2]
        
        planner = DubinsRRTStar(
            sx, sy, syaw, gx, gy, gyaw,
            vehicle_radius=config.get('vehicle_radius', 2.0),
            step_len=config.get('step_len', 30.0),
            goal_sample_rate=config.get('goal_sample_rate', 0.1),
            search_radius=config.get('search_radius', 50.0),
            iter_max=config.get('iter_max', 250)
        )
        
        planner.planning()
        
        # 从结果中提取路径
        if hasattr(planner, 'path') and planner.path:
            return [(float(p[0]), float(p[1])) for p in planner.path]
        return None
    
    def _call_bidirectional_astar_2d(self, start, goal, config):
        """调用2D双向A*算法"""
        # 将浮点坐标转换为整数坐标
        start_int = (int(round(start[0])), int(round(start[1])))
        goal_int = (int(round(goal[0])), int(round(goal[1])))
        
        try:
            from Search_based_Planning.Search_2D.Bidirectional_a_star import BidirectionalAStar
            planner = BidirectionalAStar(start_int, goal_int, config.get('heuristic_type', 'manhattan'))
            path = planner.searching()
        except (ImportError, AttributeError):
            # 如果双向A*不可用，回退到普通A*
            return self._call_astar_2d(start, goal, config)
        
        # 将整数路径坐标转换回浮点数坐标
        if path:
            path = [(float(p[0]), float(p[1])) for p in path]
        
        return path
    
    def _call_bidirectional_astar_3d(self, start, goal, config):
        """调用3D双向A*算法"""
        search_3d_path = os.path.join(os.path.dirname(__file__), 'Search_based_Planning', 'Search_3D')
        if search_3d_path not in sys.path:
            sys.path.append(search_3d_path)
        
        try:
            from Search_based_Planning.Search_3D.bidirectional_Astar3D import Weighted_A_star
            planner = Weighted_A_star(resolution=config.get('resolution', 0.5))
        except (ImportError, AttributeError):
            # 如果双向A*不可用，回退到普通A*
            return self._call_astar_3d(start, goal, config)
        
        # 设置起始点和目标点
        planner.env.start = np.array(start)
        planner.env.goal = np.array(goal)
        planner.start, planner.goal = tuple(start), tuple(goal)
        planner.x0, planner.xt = planner.start, planner.goal
        
        success = planner.run()
        if success and planner.Path:
            # 重建路径
            path_points = []
            path_points.append(planner.lastpoint)
            
            current = planner.lastpoint
            while current in planner.Parent:
                parent = planner.Parent[current]
                path_points.append(parent)
                current = parent
            
            path_points.reverse()
            return [(float(p[0]), float(p[1]), float(p[2])) for p in path_points]
        return None
    
    def _call_rrt_star_smart_2d(self, start, goal, config):
        """调用2D RRT* Smart算法"""
        try:
            from Sampling_based_Planning.rrt_2D.rrt_star_smart import RrtStarSmart
            planner = RrtStarSmart(
                x_start=start,
                x_goal=goal,
                step_len=config.get('step_len', 0.5),
                goal_sample_rate=config.get('goal_sample_rate', 0.1),
                search_radius=config.get('search_radius', 1.0),
                iter_max=config.get('iter_max', 2000)
            )
            path = planner.planning()
            return path
        except (ImportError, AttributeError):
            # 如果RRT* Smart不可用，回退到普通RRT*
            return self._call_rrt_star_2d(start, goal, config)
    
    def _call_abit_star_3d(self, start, goal, config):
        """调用3D ABIT*算法"""
        try:
            from Sampling_based_Planning.rrt_3D.ABIT_star3D import BIT_star
            planner = BIT_star()
            planner.maxiter = config.get('max_iter', 1000)
            
            # 设置起始点和目标点
            planner.env.start = np.array(start)
            planner.env.goal = np.array(goal)
            planner.x0, planner.xt = tuple(start), tuple(goal)
            
            planner.run()
            
            if planner.done and planner.Path:
                path = [(float(p[0]), float(p[1]), float(p[2])) for p in planner.Path]
                return path
        except (ImportError, AttributeError):
            # 如果ABIT*不可用，回退到RRT-Connect 3D
            return self._call_rrt_connect_3d(start, goal, config)
        return None
    
    def _call_bezier_path(self, start, goal, config):
        """调用贝塞尔路径生成器 - 先用A*找到路径点，再用贝塞尔曲线连接"""
        try:
            from CurvesGenerator.bezier_path import calc_4points_bezier_path
            import numpy as np
            
            # 首先使用A*搜索找到路径点
            astar_config = {'heuristic_type': 'manhattan', 'resolution': 1.0}
            waypoints = self._call_astar_2d(start, goal, astar_config)
            
            if not waypoints or len(waypoints) < 2:
                print("A*搜索失败，回退到直接贝塞尔路径")
                # 回退到直接计算贝塞尔路径
                return self._call_direct_bezier_path(start, goal, config)
            
            # 如果路径点太多，进行采样以减少计算量
            if len(waypoints) > 10:
                # 等间距采样，保留起始点和终点
                step = len(waypoints) // 8
                sampled_waypoints = [waypoints[0]]
                for i in range(step, len(waypoints) - 1, step):
                    sampled_waypoints.append(waypoints[i])
                sampled_waypoints.append(waypoints[-1])
                waypoints = sampled_waypoints
            
            # 使用贝塞尔曲线连接相邻的路径点
            full_path = []
            offset = config.get('offset', 3.0)
            
            for i in range(len(waypoints) - 1):
                start_point = waypoints[i]
                end_point = waypoints[i + 1]
                
                # 计算起始和目标的偏航角
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                
                # 计算起始点的偏航角
                if i == 0:
                    # 第一段：从起始点方向
                    syaw = np.arctan2(dy, dx) if len(start) < 3 else start[2]
                else:
                    # 中间段：从前一段的方向
                    prev_point = waypoints[i - 1]
                    syaw = np.arctan2(start_point[1] - prev_point[1], start_point[0] - prev_point[0])
                
                # 计算目标点的偏航角
                if i == len(waypoints) - 2:
                    # 最后一段：到目标点方向
                    gyaw = np.arctan2(dy, dx) if len(goal) < 3 else goal[2]
                else:
                    # 中间段：到下一段的方向
                    next_point = waypoints[i + 2]
                    gyaw = np.arctan2(next_point[1] - end_point[1], next_point[0] - end_point[0])
                
                # 生成贝塞尔曲线段
                path_segment, _ = calc_4points_bezier_path(
                    start_point[0], start_point[1], syaw,
                    end_point[0], end_point[1], gyaw,
                    offset
                )
                
                if path_segment is not None and len(path_segment) > 0:
                    # 避免重复添加连接点
                    if i == 0:
                        full_path.extend([(float(p[0]), float(p[1])) for p in path_segment])
                    else:
                        full_path.extend([(float(p[0]), float(p[1])) for p in path_segment[1:]])
            
            if full_path:
                return full_path
                
        except (ImportError, AttributeError, Exception) as e:
            print(f"贝塞尔路径生成失败: {e}")
            # 如果贝塞尔路径不可用，回退到三次样条
            return self._call_cubic_spline(start, goal, config)
        return None
    
    def _call_direct_bezier_path(self, start, goal, config):
        """直接计算贝塞尔路径（不使用A*）"""
        try:
            from CurvesGenerator.bezier_path import calc_4points_bezier_path
            import numpy as np
            
            # 计算起始和目标的默认偏航角
            dx = goal[0] - start[0]
            dy = goal[1] - start[1]
            default_yaw = np.arctan2(dy, dx)
            
            # 设置起始点和目标点的偏航角
            sx, sy = start[0], start[1]
            gx, gy = goal[0], goal[1]
            
            # 如果起始点包含偏航角信息，使用它；否则使用默认值
            if len(start) >= 3:
                syaw = start[2]
            else:
                syaw = default_yaw
                
            # 如果目标点包含偏航角信息，使用它；否则使用默认值
            if len(goal) >= 3:
                gyaw = goal[2]
            else:
                gyaw = default_yaw
            
            # 设置偏移参数
            offset = config.get('offset', 3.0)
            
            # 调用贝塞尔路径生成函数
            path, control_points = calc_4points_bezier_path(sx, sy, syaw, gx, gy, gyaw, offset)
            
            # 转换路径格式
            if path is not None and len(path) > 0:
                return [(float(p[0]), float(p[1])) for p in path]
                
        except Exception as e:
            print(f"直接贝塞尔路径生成失败: {e}")
        return None
    
    def _call_dubins_path(self, start, goal, config):
        """调用Dubins路径生成器 - 先用A*寻找路径点，然后用Dubins曲线连接"""
        try:
            from CurvesGenerator.dubins_path import calc_dubins_path
            import numpy as np
            
            # 首先使用A*算法获取路径点
            astar_waypoints = self._call_astar_2d(start, goal, {'heuristic_type': 'manhattan', 'resolution': 1.0})
            
            if not astar_waypoints or len(astar_waypoints) < 2:
                # 如果A*失败，使用直接连接的方法
                return self._call_direct_dubins_path(start, goal, config)
            
            # 使用A*路径点作为航路点，用Dubins曲线连接
            final_path = []
            
            # 计算起始和目标的默认偏航角
            dx = goal[0] - start[0]
            dy = goal[1] - start[1]
            default_yaw = np.arctan2(dy, dx)
            
            # 处理每一段路径
            for i in range(len(astar_waypoints) - 1):
                current_point = astar_waypoints[i]
                next_point = astar_waypoints[i + 1]
                
                # 设置当前点和下一点的偏航角
                if i == 0:
                    # 第一段，使用起始点的偏航角
                    if len(start) >= 3:
                        current_yaw = start[2]
                    else:
                        current_yaw = np.arctan2(next_point[1] - current_point[1], 
                                               next_point[0] - current_point[0])
                else:
                    # 中间段，根据前一点计算偏航角
                    prev_point = astar_waypoints[i - 1]
                    current_yaw = np.arctan2(current_point[1] - prev_point[1], 
                                           current_point[0] - prev_point[0])
                
                if i == len(astar_waypoints) - 2:
                    # 最后一段，使用目标点的偏航角
                    if len(goal) >= 3:
                        next_yaw = goal[2]
                    else:
                        next_yaw = default_yaw
                else:
                    # 中间段，根据下一点计算偏航角
                    if i + 2 < len(astar_waypoints):
                        next_next_point = astar_waypoints[i + 2]
                        next_yaw = np.arctan2(next_next_point[1] - next_point[1], 
                                            next_next_point[0] - next_point[0])
                    else:
                        next_yaw = np.arctan2(next_point[1] - current_point[1], 
                                            next_point[0] - current_point[0])
                
                # 生成当前段的Dubins路径
                curvature = config.get('curvature', 0.5)
                step_size = config.get('step_size', 0.1)
                
                segment_path = calc_dubins_path(
                    current_point[0], current_point[1], current_yaw,
                    next_point[0], next_point[1], next_yaw,
                    curvature, step_size
                )
                
                if segment_path and len(segment_path.x) > 0:
                    segment_points = list(zip(segment_path.x, segment_path.y))
                    
                    # 避免重复添加连接点
                    if final_path and len(final_path) > 0:
                        final_path.extend(segment_points[1:])  # 跳过第一个点避免重复
                    else:
                        final_path.extend(segment_points)
                else:
                    # 如果Dubins路径失败，直接连接点
                    if not final_path or final_path[-1] != current_point:
                        final_path.append(current_point)
                    final_path.append(next_point)
            
            return final_path if final_path else None
            
        except (ImportError, AttributeError, Exception) as e:
            print(f"Dubins路径生成失败: {e}")
            # 如果Dubins路径不可用，回退到直接方法
            return self._call_direct_dubins_path(start, goal, config)
    
    def _call_direct_dubins_path(self, start, goal, config):
        """直接调用Dubins路径生成器（不使用A*路径点）"""
        try:
            from CurvesGenerator.dubins_path import calc_dubins_path
            import numpy as np
            
            # 默认偏航角
            sx, sy, syaw = start[0], start[1], np.deg2rad(90)
            gx, gy, gyaw = goal[0], goal[1], np.deg2rad(0)
            
            # 如果起始点和目标点包含偏航角信息
            if len(start) >= 3:
                syaw = start[2]
            if len(goal) >= 3:
                gyaw = goal[2]
            
            curvature = config.get('curvature', 0.5)
            step_size = config.get('step_size', 0.1)
            
            path = calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, curvature, step_size)
            
            if path and len(path.x) > 0:
                return list(zip(path.x, path.y))
        except (ImportError, AttributeError):
            # 如果Dubins路径不可用，回退到三次样条
            return self._call_cubic_spline(start, goal, config)
        return None
    
    def _calculate_path_length(self, path: List[Tuple[float, ...]]) -> float:
        """计算路径长度"""
        if not path or len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(path)):
            p1, p2 = path[i-1], path[i]
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
            total_length += dist
        
        return total_length
    
    def compare_planners(self, 
                        start: Tuple[float, ...], 
                        goal: Tuple[float, ...], 
                        planner_types: List[PlannerType]) -> Dict[PlannerType, Dict]:
        """比较多个规划器的性能"""
        results = {}
        
        print("开始规划器性能比较...")
        print(f"起始点: {start}")
        print(f"目标点: {goal}")
        print(f"比较规划器: {[p.value for p in planner_types]}")
        print("-" * 50)
        
        for planner_type in planner_types:
            print(f"测试规划器: {planner_type.value}")
            path = self.plan_path(start, goal, planner_type)
            results[planner_type] = self.planning_results.get(planner_type, {})
            print("-" * 30)
        
        # 显示比较结果
        self._display_comparison_results(results)
        return results
    
    def _display_comparison_results(self, results: Dict[PlannerType, Dict]):
        """显示比较结果"""
        print("\n规划器性能比较结果:")
        print("=" * 80)
        print(f"{'规划器':<20} {'成功':<6} {'用时(s)':<8} {'路径点数':<8} {'路径长度':<10}")
        print("-" * 80)
        
        for planner_type, result in results.items():
            success = "✓" if result.get('success', False) else "✗"
            time_str = f"{result.get('planning_time', 0):.3f}"
            points = len(result.get('path', [])) if result.get('path') else 0
            length = f"{result.get('path_length', 0):.3f}"
            
            print(f"{planner_type.value:<20} {success:<6} {time_str:<8} {points:<8} {length:<10}")
    
    def visualize_environment_2d(self, planner_type: PlannerType, ax=None, title: str = "2D Environment"):
        """可视化2D环境障碍物"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            show_plot = True
        else:
            show_plot = False
            
        if planner_type in [PlannerType.A_STAR_2D, PlannerType.BIDIRECTIONAL_A_STAR_2D, 
                           PlannerType.CUBIC_SPLINE, PlannerType.BEZIER_PATH, PlannerType.DUBINS_PATH]:
            # Search_2D环境
            from Search_based_Planning.Search_2D.env import Env
            env = Env()
            
            # 绘制边界
            ax.add_patch(patches.Rectangle((0, 0), env.x_range, env.y_range, 
                                         linewidth=2, edgecolor='black', facecolor='none'))
            
            # 绘制障碍物点
            if hasattr(env, 'obs') and env.obs:
                obs_x = [obs[0] for obs in env.obs]
                obs_y = [obs[1] for obs in env.obs]
                ax.scatter(obs_x, obs_y, c='red', s=5, alpha=0.6, label='Obstacles')
            
            ax.set_xlim(-1, env.x_range + 1)
            ax.set_ylim(-1, env.y_range + 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f"{title}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect('equal')
            
        elif planner_type in [PlannerType.RRT_STAR_2D, PlannerType.RRT_2D, 
                             PlannerType.DUBINS_RRT_STAR_2D, PlannerType.RRT_STAR_SMART_2D]:
            # Sampling_2D环境
            from Sampling_based_Planning.rrt_2D.env import Env
            env = Env()
            
            # 绘制边界障碍物
            for boundary in env.obs_boundary:
                x, y, w, h = boundary
                rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                       edgecolor='black', facecolor='gray', alpha=0.8)
                ax.add_patch(rect)
            
            # 绘制矩形障碍物
            for rectangle in env.obs_rectangle:
                x, y, w, h = rectangle
                rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                       edgecolor='red', facecolor='red', alpha=0.6)
                ax.add_patch(rect)
            
            # 绘制圆形障碍物
            for circle in env.obs_circle:
                x, y, r = circle
                circle_patch = patches.Circle((x, y), r, linewidth=1, 
                                            edgecolor='blue', facecolor='blue', alpha=0.6)
                ax.add_patch(circle_patch)
            
            ax.set_xlim(env.x_range[0] - 2, env.x_range[1] + 2)
            ax.set_ylim(env.y_range[0] - 2, env.y_range[1] + 2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f"{title}")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # 添加图例
            boundary_patch = patches.Patch(color='gray', alpha=0.8, label='Boundary')
            rect_patch = patches.Patch(color='red', alpha=0.6, label='Rectangle Obstacles')
            circle_patch = patches.Patch(color='blue', alpha=0.6, label='Circle Obstacles')
            ax.legend(handles=[boundary_patch, rect_patch, circle_patch])
        
        if show_plot:
            plt.tight_layout()
            plt.show()
            return plt.gcf(), ax
        
        return None, ax
    
    def visualize_environment_3d(self, planner_type: PlannerType, ax=None, title: str = "3D Environment"):
        """可视化3D环境障碍物"""
        if planner_type in [PlannerType.A_STAR_3D, PlannerType.BIDIRECTIONAL_A_STAR_3D]:
            # Search_3D环境
            from Search_based_Planning.Search_3D.env3D import env
            env_3d = env()
            
        elif planner_type in [PlannerType.RRT_CONNECT_3D, PlannerType.RRT_3D, PlannerType.ABIT_STAR_3D]:
            # Sampling_3D环境
            from Sampling_based_Planning.rrt_3D.env3D import env
            env_3d = env()
        else:
            print(f"不支持的3D规划器类型: {planner_type}")
            return None, None
        
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            show_plot = True
        else:
            show_plot = False
            # 确保传入的ax已经是3D类型
            if not hasattr(ax, 'zaxis'):
                print("警告: 传入的ax不是3D轴，无法绘制3D图形")
                return None, None
        
        # 绘制工作空间边界
        boundary = env_3d.boundary
        xmin, ymin, zmin, xmax, ymax, zmax = boundary
        
        # 绘制边界框架
        # 底面
        ax.plot([xmin, xmax], [ymin, ymin], [zmin, zmin], 'k-', alpha=0.3)
        ax.plot([xmin, xmax], [ymax, ymax], [zmin, zmin], 'k-', alpha=0.3)
        ax.plot([xmin, xmin], [ymin, ymax], [zmin, zmin], 'k-', alpha=0.3)
        ax.plot([xmax, xmax], [ymin, ymax], [zmin, zmin], 'k-', alpha=0.3)
        
        # 顶面
        ax.plot([xmin, xmax], [ymin, ymin], [zmax, zmax], 'k-', alpha=0.3)
        ax.plot([xmin, xmax], [ymax, ymax], [zmax, zmax], 'k-', alpha=0.3)
        ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], 'k-', alpha=0.3)
        ax.plot([xmax, xmax], [ymin, ymax], [zmax, zmax], 'k-', alpha=0.3)
        
        # 垂直边
        ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], 'k-', alpha=0.3)
        ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], 'k-', alpha=0.3)
        ax.plot([xmin, xmin], [ymax, ymax], [zmin, zmax], 'k-', alpha=0.3)
        ax.plot([xmax, xmax], [ymax, ymax], [zmin, zmax], 'k-', alpha=0.3)
        
        # 绘制AABB障碍物（立方体）
        if hasattr(env_3d, 'blocks') and len(env_3d.blocks) > 0:
            for i, block in enumerate(env_3d.blocks):
                x1, y1, z1, x2, y2, z2 = block
                
                # 绘制立方体的12条边
                # 底面4条边
                ax.plot([x1, x2], [y1, y1], [z1, z1], 'r-', linewidth=2)
                ax.plot([x1, x2], [y2, y2], [z1, z1], 'r-', linewidth=2)
                ax.plot([x1, x1], [y1, y2], [z1, z1], 'r-', linewidth=2)
                ax.plot([x2, x2], [y1, y2], [z1, z1], 'r-', linewidth=2)
                
                # 顶面4条边
                ax.plot([x1, x2], [y1, y1], [z2, z2], 'r-', linewidth=2)
                ax.plot([x1, x2], [y2, y2], [z2, z2], 'r-', linewidth=2)
                ax.plot([x1, x1], [y1, y2], [z2, z2], 'r-', linewidth=2)
                ax.plot([x2, x2], [y1, y2], [z2, z2], 'r-', linewidth=2)
                
                # 垂直4条边
                ax.plot([x1, x1], [y1, y1], [z1, z2], 'r-', linewidth=2)
                ax.plot([x2, x2], [y1, y1], [z1, z2], 'r-', linewidth=2)
                ax.plot([x1, x1], [y2, y2], [z1, z2], 'r-', linewidth=2)
                ax.plot([x2, x2], [y2, y2], [z1, z2], 'r-', linewidth=2)
        
        # 绘制球体障碍物
        if hasattr(env_3d, 'balls') and len(env_3d.balls) > 0:
            for ball in env_3d.balls:
                cx, cy, cz, r = ball
                
                # 创建球面
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x_sphere = cx + r * np.outer(np.cos(u), np.sin(v))
                y_sphere = cy + r * np.outer(np.sin(u), np.sin(v))
                z_sphere = cz + r * np.outer(np.ones(np.size(u)), np.cos(v))
                
                ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='blue')
        
        # 绘制起始点和目标点
        if hasattr(env_3d, 'start'):
            ax.scatter(env_3d.start[0], env_3d.start[1], env_3d.start[2], 
                      c='green', s=100, label='Start', marker='o')
        
        if hasattr(env_3d, 'goal'):
            ax.scatter(env_3d.goal[0], env_3d.goal[1], env_3d.goal[2], 
                      c='red', s=100, label='Goal', marker='s')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{title}")
        ax.legend()
        
        # 设置相等的坐标轴比例
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2.0
        mid_x = (xmax + xmin) * 0.5
        mid_y = (ymax + ymin) * 0.5
        mid_z = (zmax + zmin) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if show_plot:
            plt.tight_layout()
            plt.show()
            return plt.gcf(), ax
        
        return None, ax
    
    def visualize_path_with_environment(self, 
                                      path: List[Tuple[float, ...]], 
                                      start: Tuple[float, ...], 
                                      goal: Tuple[float, ...],
                                      planner_type: PlannerType,
                                      ax=None,
                                      title: str = "Path Planning Result"):
        """可视化路径和环境"""
        if not path:
            print("没有路径可以可视化")
            return None, None
        
        if len(path[0]) == 2:  # 2D路径
            _, ax = self.visualize_environment_2d(planner_type, ax, title)
            
            # 绘制路径
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            ax.plot(x_coords, y_coords, 'g-', linewidth=3, label='Path', alpha=0.8)
            
            # 绘制起始点和目标点
            ax.plot(start[0], start[1], 'go', markersize=12, label='Start', markeredgecolor='black')
            ax.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal', markeredgecolor='black')
            
            ax.legend()
            
        elif len(path[0]) == 3:  # 3D路径
            _, ax = self.visualize_environment_3d(planner_type, ax, title)
            
            # 绘制路径
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            z_coords = [p[2] for p in path]
            ax.plot(x_coords, y_coords, z_coords, 'g-', linewidth=3, label='Path', alpha=0.8)
            
            # 重新绘制起始点和目标点（覆盖环境中的点）
            ax.scatter(start[0], start[1], start[2], c='green', s=150, label='Start', 
                      marker='o', edgecolors='black', linewidth=2)
            ax.scatter(goal[0], goal[1], goal[2], c='red', s=150, label='Goal', 
                      marker='s', edgecolors='black', linewidth=2)
            
            ax.legend()
        
        return None, ax
    
    def get_available_planners(self) -> Dict[PlannerCategory, List[PlannerType]]:
        """获取可用的规划器列表，按类别分组"""
        planners = {
            PlannerCategory.SEARCH_BASED: [
                PlannerType.A_STAR_2D,
                PlannerType.A_STAR_3D,
                PlannerType.BIDIRECTIONAL_A_STAR_2D,
                PlannerType.BIDIRECTIONAL_A_STAR_3D,
            ],
            PlannerCategory.SAMPLING_BASED: [
                PlannerType.RRT_STAR_2D,
                PlannerType.DUBINS_RRT_STAR_2D,
                PlannerType.RRT_CONNECT_3D,
                PlannerType.RRT_STAR_SMART_2D,
                PlannerType.ABIT_STAR_3D,
            ],
            PlannerCategory.CURVE_GENERATOR: [
                PlannerType.CUBIC_SPLINE,
                PlannerType.BEZIER_PATH,
                PlannerType.DUBINS_PATH,
            ]
        }
        return planners
    
    def get_planning_results(self) -> Dict[PlannerType, Dict]:
        """获取所有规划结果"""
        return self.planning_results.copy()

# 使用示例
if __name__ == "__main__":
    # 创建路径规划管理器
    manager = PathPlannerManager()
    
    print("=== 统一环境可视化演示 ===")
    
    # 创建一个大的figure来显示所有结果
    fig = plt.figure(figsize=(20, 12))
    
    # 2D路径规划示例
    start_2d = (5.0, 5.0)
    goal_2d = (45.0, 25.0)
    start_3d = (2.0, 2.0, 2.0)
    goal_3d = (18.0, 18.0, 4.0)
    
    
    path_astar_2d = manager.plan_path(start_2d, goal_2d, PlannerType.A_STAR_2D)
    if path_astar_2d:
        ax1 = fig.add_subplot(2, 3, 1)
        manager.visualize_path_with_environment(path_astar_2d, start_2d, goal_2d, 
                                              PlannerType.A_STAR_2D, ax1, "A* 2D Path Planning")
    
    path_rrt_star_2d = manager.plan_path(start_2d, goal_2d, PlannerType.RRT_STAR_2D)
    if path_rrt_star_2d:
        ax2 = fig.add_subplot(2, 3, 2)
        manager.visualize_path_with_environment(path_rrt_star_2d, start_2d, goal_2d, 
                                              PlannerType.RRT_STAR_2D, ax2, "RRT* 2D Path Planning")
    path_cubic_spline = manager.plan_path(start_2d, goal_2d, PlannerType.CUBIC_SPLINE)
    if path_cubic_spline:
        ax3 = fig.add_subplot(2, 3, 3)
        manager.visualize_path_with_environment(path_cubic_spline, start_2d, goal_2d, 
                                              PlannerType.CUBIC_SPLINE, ax3, "Cubic Spline Path Planning")
    path_dubins_path = manager.plan_path(start_2d, goal_2d, PlannerType.DUBINS_PATH)
    if path_dubins_path:
        ax4 = fig.add_subplot(2, 3, 4)
        manager.visualize_path_with_environment(path_dubins_path, start_2d, goal_2d, 
                                              PlannerType.DUBINS_PATH, ax4, "Dubins Path Planning")
    path_bezier_path = manager.plan_path(start_2d, goal_2d, PlannerType.BEZIER_PATH)
    if path_bezier_path:
        ax5 = fig.add_subplot(2, 3, 5)
        manager.visualize_path_with_environment(path_bezier_path, start_2d, goal_2d, 
                                              PlannerType.BEZIER_PATH, ax5, "Bezier Path Planning")
    path_smart_rrt_star_2d = manager.plan_path(start_2d, goal_2d, PlannerType.RRT_STAR_SMART_2D)
    if path_smart_rrt_star_2d:
        ax6 = fig.add_subplot(2, 3, 6)
        manager.visualize_path_with_environment(path_smart_rrt_star_2d, start_2d, goal_2d, 
                                              PlannerType.RRT_STAR_SMART_2D, ax6, "Smart RRT* 2D Path Planning")
    # path_astar_3d = manager.plan_path(start_3d, goal_3d, PlannerType.A_STAR_3D)
    # if path_astar_3d:
    #     print(f"A* 3D规划成功，路径包含 {len(path_astar_3d)} 个点")
    #     ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    #     manager.visualize_path_with_environment(path_astar_3d, start_3d, goal_3d, 
    #                                           PlannerType.A_STAR_3D, ax4, "A* 3D Path Planning")
    # path_rrt_star_3d = manager.plan_path(start_3d, goal_3d, PlannerType.RRT_CONNECT_3D)
    # if path_rrt_star_3d:
    #     print(f"RRT-Connect 3D规划成功，路径包含 {len(path_rrt_star_3d)} 个点")
    #     ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    #     manager.visualize_path_with_environment(path_rrt_star_3d, start_3d, goal_3d, 
    #                                           PlannerType.RRT_CONNECT_3D, ax5, "RRT-Connect 3D Path Planning")
    
    # 调整子图间距并显示
    # plt.tight_layout()
    # plt.tight_layout(pad=2.0)
    
    # 保存图形到文件
    plt.savefig('/home/yyf/PathPlanning/unified_planning_demo.png', dpi=300, bbox_inches='tight')
    print("图形已保存到: /home/yyf/PathPlanning/unified_planning_demo.png")
    
    plt.show()
    # # 演示规划器性能比较
    # print("\n=== 规划器性能比较 ===")
    
    # # 2D规划器比较
    # print("\n--- 2D规划器比较 ---")
    # planners_2d = [PlannerType.A_STAR_2D, PlannerType.CUBIC_SPLINE]
    # comparison_2d = manager.compare_planners(start_2d, goal_2d, planners_2d)
    
    # # 3D规划器比较  
    # print("\n--- 3D规划器比较 ---")
    # planners_3d = [PlannerType.A_STAR_3D]
    # comparison_3d = manager.compare_planners(start_3d, goal_3d, planners_3d)
    
    # 显示可用的规划器
    exit(0)
    from planner_manager import PathPlannerManager, PlannerType
    manager = PathPlannerManager()
    available_planners = manager.get_available_planners()
    for category, planners in available_planners.items():
        print(f"{category.value}:")
        for planner in planners:
            info = manager.get_planner_info(planner)
            print(f"  - {info['name']} ({planner.value}): {info['description']}")
    
